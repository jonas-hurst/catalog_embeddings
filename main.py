import os
import glob
import json
import concurrent.futures
from datetime import datetime, timezone
import shutil

from affine import Affine
import rasterio
import numpy as np
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box, mapping, Polygon
import pystac


def process_directory(args):
    """
    Worker function to process a single directory of GeoTIFFs.
    Returns a dictionary of metadata for STAC generation on success.
    """

    def embeddings_to_arrow(stacked: np.ndarray) -> pa.Array:
        """stacked: (N, H, W, C) contiguous float32 -> FixedSizeList<FixedSizeList<FixedSizeList<float>>>"""
        assert stacked.dtype == np.float32 and stacked.flags["C_CONTIGUOUS"]
        n, h, w, c = stacked.shape

        flat = pa.array(stacked.reshape(-1))  # zero-copy FloatArray
        lvl_c = pa.FixedSizeListArray.from_arrays(flat, c)  # (N*H*W) lists of C
        lvl_w = pa.FixedSizeListArray.from_arrays(lvl_c, w)  # (N*H)   lists of W
        lvl_h = pa.FixedSizeListArray.from_arrays(lvl_w, h)  # (N)     lists of H
        return lvl_h

    # -------------------------------------------------------------------------------

    input_dir: str = args[0]
    output_file: str = args[1]
    stac_out_dir: str = args[2]
    collection_id = args[3]
    px_scale_factor = args[4]
    item_id = os.path.basename(input_dir)
    if item_id.endswith(".SAFE"):
        item_id = item_id[:-5]  # remove .SAFE
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*_embedding.tif")))

    if not tif_files:
        return {"status": "skipped", "message": f"No .tif files found in {input_dir}"}

    ids, geoms = [], []
    stacked = None
    native_crs = None

    try:
        # 1. Read files and extract data
        for i, f in enumerate(tif_files):
            with rasterio.open(f) as src:
                a = src.read()  # (C, H, W)
                if stacked is None:  # derive shape from the first file
                    c, h, w = a.shape
                    stacked = np.empty((len(tif_files), h, w, c), dtype=np.float32)
                    native_crs = src.crs
                elif a.shape != (c, h, w):
                    raise ValueError(f"{f}: shape {a.shape} != {(c, h, w)}")
                elif src.crs != native_crs:
                    raise ValueError(f"{f}: CRS {src.crs} != {native_crs}")

                # single copy: transpose + cast written directly into the output buffer
                stacked[i] = np.transpose(a, (1, 2, 0))

                # fix pixel size metadata and build footprint polygon
                transform = src.transform * Affine.scale(px_scale_factor, px_scale_factor)
                width, height = src.width, src.height
                footprint = Polygon([
                    transform * (0, 0),
                    transform * (width, 0),
                    transform * (width, height),
                    transform * (0, height),
                ])

                ids.append(os.path.basename(f))
                geoms.append(footprint)

        # geometry-only GeoDataFrame: the embeddings never need to go through pandas
        gdf = gpd.GeoDataFrame({"ID": ids}, geometry=geoms, crs=native_crs)
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")

        table = pa.Table.from_arrays(
            [pa.array(gdf["ID"], type=pa.string()),
             pa.array(gdf.geometry.to_wkb()),
             embeddings_to_arrow(stacked)],
            names=["ID", "geometry", "embedding"],
        )

        # 5. Add standard GeoParquet Metadata
        bbox = list(gdf.total_bounds)
        geo_metadata = {
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["Polygon"],
                    "crs": gdf.crs.to_json_dict(),
                    "bbox": bbox
                }
            }
        }

        custom_metadata = table.schema.metadata or {}
        custom_metadata[b"geo"] = json.dumps(geo_metadata).encode("utf-8")
        table = table.replace_schema_metadata(custom_metadata)

        # 6. Ensure output directory exists and write to disk
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pq.write_table(table, output_file)

        # 7. Build STAC Item

        if item_id.startswith("S2"):
            source_url = "stac.dataspace.copernicus.eu/v1"
            source_collection = "sentinel-2-l2a"
            stac_datetime = datetime.strptime(item_id.split("_")[2], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            data_key = "sentinel_2_embeddings"
        elif item_id.startswith("S1"):
            source_url = "stac.terrabyte.lrz.de/public/api"
            source_collection = "sentinel-1-grd"
            stac_datetime = datetime.strptime(item_id.split("_")[4], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            data_key = "sentinel_1_embeddings"
        else:
            raise Exception(f"Item type {item_id} unsupported")

        item = pystac.Item(
            id=item_id,
            geometry=mapping(box(*bbox)),
            bbox=bbox,
            datetime=stac_datetime,
            properties={"proj:code": "EPSG:4326"},
            collection=collection_id,
            stac_extensions=[
                "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
                "https://stac-extensions.github.io/embeddings/v0.0.1/schema.json"
            ]
        )

        item.add_asset(
            "embeddings",
            pystac.Asset(
                href=f"https://e2s.s3.eu-de.cloud-object-storage.appdomain.cloud/embeddings/maritime/{data_key}/{output_file}",
                media_type="application/vnd.apache.parquet",
                roles=["embeddings"],
                title=f"TerraMind Embeddings",
                description="The embeddings as geoparquet"
            )
        )


        item.add_link(
            pystac.Link(
                "emb:source-data",
                f"https://{source_url}/collections/{source_collection}/items/{item_id}",
                "application/json"
            )
        )

        item.save_object(False, f"{stac_out_dir}/{item_id}.json")

        return {
            "status": "success",
            "id": item_id,
            "parquet_path": output_file,
            "bbox": bbox,
            "datetime": stac_datetime.isoformat(),
            "count": len(gdf)
        }

    except Exception as e:
        return {"status": "error", "message": f"Error in {input_dir}: {str(e)}"}


def process_and_catalog(parent_input_dir, parent_output_dir, stac_out_dir, collection_id):
    """
    Orchestrates the parallel processing of TIFFs and the creation of the STAC catalog.
    """
    os.makedirs(parent_output_dir, exist_ok=True)
    os.makedirs(stac_out_dir, exist_ok=True)

    # 1. Build Task List
    tasks = []
    for entry in os.scandir(parent_input_dir):
        if entry.is_dir():
            input_dir = entry.path
            output_file = os.path.join(parent_output_dir, f"{entry.name}.parquet")
            tasks.append((input_dir, output_file, stac_out_dir, collection_id, 16))

    if not tasks:
        print(f"No subdirectories found in {parent_input_dir}")
        return

    print(f"Found {len(tasks)} directories. Processing in parallel...")
    successful_results = []

    # 2. Process in Parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_directory, tasks)

        for result in results:
            if result["status"] == "success":
                print(f"Success: Processed {result['count']} files for {result['id']}")
                successful_results.append(result)
            else:
                print(result["message"])

    if not successful_results:
        print("No successful outputs to catalog. Exiting.")
        return

    # 3. Build STAC Items
    print(f"Building STAC Catalog for {len(successful_results)} items...")

    all_bboxes = []
    all_datetimes = []

    for res in successful_results:
        bbox = res["bbox"]
        all_bboxes.append(bbox)
        all_datetimes.append(datetime.fromisoformat(res["datetime"]))

    # 4. Calculate overall Extents and build STAC Collection
    minx = min(b[0] for b in all_bboxes)
    miny = min(b[1] for b in all_bboxes)
    maxx = max(b[2] for b in all_bboxes)
    maxy = max(b[3] for b in all_bboxes)

    spatial_extent = pystac.SpatialExtent(bboxes=[[minx, miny, maxx, maxy]])
    temporal_extent = pystac.TemporalExtent(
        intervals=[[min(all_datetimes), max(all_datetimes)]])
    extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

    collection = pystac.Collection(
        id=collection_id,
        title=collection_id,
        description="A STAC Collection of Sentinel embeddings, produced with the TerraMind Tokenizer.",
        stac_extensions=[
            "https://stac-extensions.github.io/embeddings/v0.0.1/schema.json",
            "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
        ],
        extent=extent,
        license="CC-BY-4.0",
        providers=[
            pystac.Provider(
                "Embed2Scale",
                "EU Horizon Embed2Scale Project, GA Number 101131841",
                [pystac.ProviderRole.PRODUCER, pystac.ProviderRole.PROCESSOR],
                "https://embed2scale.eu")
        ],
        keywords=["terramind", "embeddings"],
        extra_fields={
            "data_type": "float32",
            "emb:type": "patch",
            "emb:dimensions": 5,
            "emb:chip_layout": {"layout_type": "regular_grid"},
            "proj:code": "EPSG:4326",
        },
    )

    collection.item_assets = {
        "embeddings": pystac.ItemAssetDefinition.create(
            media_type="application/vnd.apache.parquet",
            roles=["embeddings"],
            title=f"TerraMind Embeddings",
            description="The embeddings as geoparquet"
        )
    }

    # 5. Save STAC structure
    collection.save_object(False, f"{stac_out_dir}/collection.json")

    print(f"Process complete. STAC Collection saved to {os.path.abspath(stac_out_dir)}")


if __name__ == "__main__":
    # --- Configuration ---
    # Directory containing subdirectories of TIFFs
    S2_RAW_DATA_DIR = "/dss/dsshome1/04/di97bis/e2s/maritime/new/03_interim_data/02_terramind_embeddings/02_S2_embeddings"
    S2_OUT_DIR = "/dss/dsshome1/04/di97bis/e2s/maritime/99_preprocess/s2"
    S2_PARQUETS_DIR = f"{S2_OUT_DIR}/pq"  # Directory to store the output GeoParquet files
    S2_STAC_DIR = f"{S2_OUT_DIR}/stac"  # Directory to store the generated STAC Collection and Items

    if os.path.exists(S2_OUT_DIR):
        shutil.rmtree(S2_OUT_DIR)

    process_and_catalog(S2_RAW_DATA_DIR, S2_PARQUETS_DIR, S2_STAC_DIR, "TerraMind-Sentinel-2-tokenizer-embeddings")

    S1_RAW_DATA_DIR = "/dss/dsshome1/04/di97bis/e2s/maritime/new/03_interim_data/02_terramind_embeddings/01_S1_embeddings"
    S1_OUT_DIR = "/dss/dsshome1/04/di97bis/e2s/maritime/99_preprocess/s1"
    S1_PARQUETS_DIR = f"{S1_OUT_DIR}/pq"  # Directory to store the output GeoParquet files
    S1_STAC_DIR = f"{S1_OUT_DIR}/stac"  # Directory to store the generated STAC Collection and Items

    if os.path.exists(S1_OUT_DIR):
        shutil.rmtree(S1_OUT_DIR)

    process_and_catalog(S1_RAW_DATA_DIR, S1_PARQUETS_DIR, S1_STAC_DIR, "TerraMind-Sentinel-1-tokenizer-embeddings")