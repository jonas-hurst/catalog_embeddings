import os
import glob
import json
import concurrent.futures
from datetime import datetime, timezone

import rasterio
import numpy as np
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box, mapping
import pystac


def process_directory(args):
    """
    Worker function to process a single directory of GeoTIFFs.
    Returns a dictionary of metadata for STAC generation on success.
    """
    input_dir, output_file = args
    item_id = os.path.basename(input_dir)
    records = []
    native_crs = None

    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    if not tif_files:
        return {"status": "skipped", "message": f"No .tif files found in {input_dir}"}

    try:
        # 1. Read files and extract data
        for f in tif_files:
            with rasterio.open(f) as src:
                arr = src.read()
                arr = np.transpose(arr, (1, 2, 0))  # (14, 14, 5)

                geom = box(*src.bounds)

                if native_crs is None:
                    native_crs = src.crs

                records.append({
                    "ID": os.path.basename(f),
                    "geometry": geom,
                    "embedding": arr.tolist()
                })

        # 2. Initialize GeoDataFrame and reproject to EPSG:4326
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=native_crs)

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")

        # 3. Define the nested FixedSizeList PyArrow type for (14, 14, 5)
        type_inner = pa.fixed_size_list(pa.float32(), 5)
        type_mid = pa.fixed_size_list(type_inner, 14)
        type_outer = pa.fixed_size_list(type_mid, 14)

        embedding_array = pa.array(gdf["embedding"], type=type_outer)

        # 4. Construct PyArrow Table
        wkb_geometry = pa.array(gdf.geometry.to_wkb())
        id_array = pa.array(gdf["ID"], type=pa.string())

        table = pa.Table.from_arrays(
            [id_array, wkb_geometry, embedding_array],
            names=["ID", "geometry", "embedding"]
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

        return {
            "status": "success",
            "id": item_id,
            "parquet_path": output_file,
            "bbox": bbox,
            "count": len(gdf)
        }

    except Exception as e:
        return {"status": "error", "message": f"Error in {input_dir}: {str(e)}"}


def process_and_catalog(parent_input_dir, parent_output_dir, stac_out_dir):
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
            tasks.append((input_dir, output_file))

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
    collection_time = datetime.now(timezone.utc)
    stac_items = []
    all_bboxes = []

    for res in successful_results:
        bbox = res["bbox"]
        all_bboxes.append(bbox)

        # Geometry for the item is the bounding box of the parquet file
        geom = mapping(box(*bbox))

        item = pystac.Item(
            id=res["id"],
            geometry=geom,
            bbox=bbox,
            datetime=collection_time,
            properties={"proj:epsg": 4326}
        )

        # Path relative to the final STAC directory
        rel_path = os.path.relpath(res["parquet_path"], stac_out_dir)

        item.add_asset(
            "data",
            pystac.Asset(
                href=rel_path,
                media_type="application/vnd.apache.parquet",
                roles=["data"],
                title=f"ViT Embeddings for {res['id']}"
            )
        )
        stac_items.append(item)

    # 4. Calculate overall Extents and build STAC Collection
    minx = min(b[0] for b in all_bboxes)
    miny = min(b[1] for b in all_bboxes)
    maxx = max(b[2] for b in all_bboxes)
    maxy = max(b[3] for b in all_bboxes)

    spatial_extent = pystac.SpatialExtent(bboxes=[[minx, miny, maxx, maxy]])
    temporal_extent = pystac.TemporalExtent(
        intervals=[[collection_time, collection_time]])
    extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

    collection = pystac.Collection(
        id="vit-embeddings-collection",
        title="Vision Transformer Embeddings",
        description="A STAC Collection of GeoParquet files containing (14, 14, 5) dense embeddings.",
        extent=extent,
        license="proprietary"
    )

    for item in stac_items:
        collection.add_item(item)

    # 5. Save STAC structure
    collection.normalize_hrefs(stac_out_dir)
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    print(f"Process complete. STAC Collection saved to {os.path.abspath(stac_out_dir)}")


if __name__ == "__main__":
    # --- Configuration ---
    # Directory containing subdirectories of TIFFs
    RAW_DATA_DIR = "./data/raw_tiffs"

    # Directory to store the output GeoParquet files
    PARQUETS_DIR = "./data/processed_parquets"

    # Directory to store the generated STAC Collection and Items
    STAC_DIR = "./data/stac_catalog"

    process_and_catalog(RAW_DATA_DIR, PARQUETS_DIR, STAC_DIR)