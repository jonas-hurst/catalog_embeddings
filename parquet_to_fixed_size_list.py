#!/usr/bin/env python3
"""
Convert variable-length list columns in a Parquet file to FixedSizeList columns.

Examples
--------
# auto-detect every list column and infer its length
python parquet_to_fixed_size_list.py in.parquet out.parquet

# only look at some columns, and pin the size (skips the inference scan)
python parquet_to_fixed_size_list.py in.parquet out.parquet -c embedding -s embedding=768

# nested list<list<float>> -> fixed_size_list<fixed_size_list<float>[4]>[3]
python parquet_to_fixed_size_list.py in.parquet out.parquet -s matrix=3,4

# just report what would happen
python parquet_to_fixed_size_list.py in.parquet --inspect
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def eprint(*a, **kw):
    print(*a, file=sys.stderr, **kw)


def is_var_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def fixed_size_list(value_type, size: int) -> pa.DataType:
    """pa.list_(value_type, N) is how you build a FixedSizeListType.
    (There is no pa.fixed_size_list.) Accepts a DataType or a Field."""
    return pa.list_(value_type, size)


# --------------------------------------------------------------------------- #
# 1. find candidate list paths in the schema
# --------------------------------------------------------------------------- #
def collect_list_paths(t: pa.DataType, path: str, acc: List[str],
                       top_level_only: bool) -> None:
    """Collect dotted paths of variable-length list nodes. Nested list levels get '.list'."""
    if is_var_list(t):
        acc.append(path)
        if not top_level_only:
            collect_list_paths(t.value_type, path + ".list", acc, top_level_only)
    elif pa.types.is_fixed_size_list(t):
        if not top_level_only:
            collect_list_paths(t.value_type, path + ".list", acc, top_level_only)
    elif pa.types.is_struct(t):
        for f in t:
            collect_list_paths(f.type, f"{path}.{f.name}", acc, top_level_only)
    # maps / unions are intentionally left alone


# --------------------------------------------------------------------------- #
# 2. infer list lengths
# --------------------------------------------------------------------------- #
def _stats_update(stats: Dict[str, dict], path: str, arr: pa.Array) -> None:
    lens = pc.drop_null(pc.list_value_length(arr))
    e = stats.setdefault(path, {"min": None, "max": None, "n": 0})
    if len(lens) == 0:
        return
    mn, mx = pc.min(lens).as_py(), pc.max(lens).as_py()
    e["min"] = mn if e["min"] is None else min(e["min"], mn)
    e["max"] = mx if e["max"] is None else max(e["max"], mx)
    e["n"] += len(lens)


def collect_stats(arr, path: str, stats: Dict[str, dict]) -> None:
    if isinstance(arr, pa.ChunkedArray):
        for c in arr.chunks:
            collect_stats(c, path, stats)
        return
    t = arr.type
    if is_var_list(t):
        _stats_update(stats, path, arr)
        collect_stats(arr.flatten(), path + ".list", stats)
    elif pa.types.is_fixed_size_list(t):
        collect_stats(arr.flatten(), path + ".list", stats)
    elif pa.types.is_struct(t):
        for i, f in enumerate(t):
            collect_stats(arr.field(i), f"{path}.{f.name}", stats)


# --------------------------------------------------------------------------- #
# 3. conversion
# --------------------------------------------------------------------------- #
def _to_fixed_size_list(arr: pa.Array, size: int, path: str,
                        sizes: Dict[str, int]) -> pa.Array:
    """list<T>/large_list<T> -> fixed_size_list<T>[size], preserving top-level nulls."""
    n = len(arr)
    offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    valid = (
        np.ones(n, dtype=bool)
        if arr.null_count == 0
        else arr.is_valid().to_numpy(zero_copy_only=False)
    )

    starts = offsets[:n]
    lengths = offsets[1: n + 1] - starts
    bad = valid & (lengths != size)
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(
            f"'{path}': found a list of length {int(lengths[i])} (row {i} of this batch) "
            f"but target fixed size is {size}. Use --size to override or --skip-ragged."
        )

    if size > 0:
        idx = (starts[:, None] + np.arange(size, dtype=np.int64)[None, :]).reshape(-1)
        null_slots = np.repeat(~valid, size)
        if null_slots.any():
            idx = np.where(null_slots, 0, idx)  # dummy index; masked out anyway
            indices = pa.array(idx, type=pa.int64(), mask=null_slots)
        else:
            indices = pa.array(idx, type=pa.int64())
        child = arr.values.take(indices)
    else:
        child = arr.values.slice(0, 0)

    # recurse (handles list<list<...>>)
    child = convert_array(child, path + ".list", sizes)

    out_type = fixed_size_list(child.type, size)
    if valid.all() and size > 0:
        return pa.FixedSizeListArray.from_arrays(child, size)

    validity_buf = None
    null_count = 0
    if not valid.all():
        validity_buf = pa.array(valid, type=pa.bool_()).buffers()[1]
        null_count = int((~valid).sum())
    return pa.Array.from_buffers(
        out_type, n, [validity_buf], null_count=null_count, children=[child]
    )


def convert_array(arr, path: str, sizes: Dict[str, int]):
    if isinstance(arr, pa.ChunkedArray):
        if arr.num_chunks == 0:
            empty = convert_array(pa.array([], type=arr.type), path, sizes)
            return pa.chunked_array([], type=empty.type)
        return pa.chunked_array([convert_array(c, path, sizes) for c in arr.chunks])

    t = arr.type

    if is_var_list(t):
        size = sizes.get(path)
        if size is not None:
            return _to_fixed_size_list(arr, size, path, sizes)
        # not converting this level, but maybe a deeper one
        new_child = convert_array(arr.values, path + ".list", sizes)
        if new_child.type == t.value_type:
            return arr
        vf = t.value_field
        new_field = pa.field(vf.name, new_child.type, vf.nullable)
        new_type = pa.list_(new_field) if pa.types.is_list(t) else pa.large_list(
            new_field)
        return pa.Array.from_buffers(
            new_type, len(arr), arr.buffers()[:2],
            null_count=arr.null_count, offset=arr.offset, children=[new_child],
        )

    if pa.types.is_fixed_size_list(t):
        new_child = convert_array(arr.values, path + ".list", sizes)
        if new_child.type == t.value_type:
            return arr
        vf = t.value_field
        new_type = fixed_size_list(
            pa.field(vf.name, new_child.type, vf.nullable), t.list_size
        )
        return pa.Array.from_buffers(
            new_type, len(arr), arr.buffers()[:1],
            null_count=arr.null_count, offset=arr.offset, children=[new_child],
        )

    if pa.types.is_struct(t):
        children, fields, changed = [], [], False
        for i, f in enumerate(t):
            c = convert_array(arr.field(i), f"{path}.{f.name}", sizes)
            changed |= c.type != f.type
            children.append(c)
            fields.append(pa.field(f.name, c.type, f.nullable, f.metadata))
        if not changed:
            return arr
        mask = pc.is_null(arr) if arr.null_count else None
        return pa.StructArray.from_arrays(children, fields=fields, mask=mask)

    return arr


def convert_batch(batch: pa.RecordBatch, sizes: Dict[str, int], columns) -> List[
    pa.Array]:
    out = []
    for i, f in enumerate(batch.schema):
        a = batch.column(i)
        out.append(convert_array(a, f.name, sizes) if f.name in columns else a)
    return out


def output_schema(in_schema: pa.Schema, sizes, columns,
                  keep_metadata=True) -> pa.Schema:
    """Derive the output schema by converting an empty batch (guarantees an exact match)."""
    arrays = [pa.array([], type=f.type) for f in in_schema]
    empty = pa.RecordBatch.from_arrays(arrays, schema=in_schema)
    converted = convert_batch(empty, sizes, columns)
    schema = pa.schema(
        [pa.field(f.name, a.type, f.nullable, f.metadata)
         for f, a in zip(in_schema, converted)]
    )
    return schema.with_metadata(in_schema.metadata) if keep_metadata else schema


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def parse_size_specs(specs: List[str]) -> Tuple[Dict[str, int], Optional[int]]:
    explicit: Dict[str, int] = {}
    default: Optional[int] = None
    for s in specs:
        if "=" in s:
            path, val = s.rsplit("=", 1)
            path = path.strip()
            for depth, v in enumerate(int(x) for x in val.split(",")):
                explicit[path + ".list" * depth] = v
        else:
            default = int(s)
    return explicit, default


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rewrite a Parquet file converting list columns to FixedSizeList columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input")
    ap.add_argument("output", nargs="?",
                    help="output parquet path (omit with --inspect)")
    ap.add_argument("-c", "--columns", nargs="+",
                    help="only convert these top-level columns")
    ap.add_argument("-s", "--size", action="append", default=[], metavar="PATH=N",
                    help="pin a size: 'col=768', nested: 'col=3,4', or bare 'N' as default")
    ap.add_argument("--top-level-only", action="store_true",
                    help="only convert the outermost list level of each column")
    ap.add_argument("--skip-ragged", action="store_true",
                    help="leave columns whose lists vary in length as-is instead of erroring")
    ap.add_argument("--inspect", action="store_true",
                    help="report inferred sizes and exit")
    ap.add_argument("--infer-rows", type=int, default=None,
                    help="stop the inference scan after N rows (default: scan all)")
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--row-group-size", type=int, default=None,
                    help="rows per output row group (default: --batch-size)")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--compression-level", type=int, default=None)
    ap.add_argument("--drop-metadata", action="store_true",
                    help="do not copy the input schema key/value metadata (e.g. stale pandas info)")
    args = ap.parse_args()

    if not args.inspect and not args.output:
        ap.error("output path is required (or use --inspect)")
    if args.output and args.output == args.input:
        ap.error("refusing to overwrite the input file")

    pf = pq.ParquetFile(args.input)
    in_schema = pf.schema_arrow

    selected = list(args.columns) if args.columns else [f.name for f in in_schema]
    missing = [c for c in selected if c not in in_schema.names]
    if missing:
        ap.error(f"column(s) not in file: {missing}")

    candidates: List[str] = []
    for name in selected:
        collect_list_paths(in_schema.field(name).type, name, candidates,
                           args.top_level_only)

    if not candidates:
        eprint("No variable-length list columns found; nothing to convert.")
        if args.inspect:
            return 0

    explicit, default_size = parse_size_specs(args.size)
    for p in explicit:
        if p not in candidates:
            eprint(f"warning: --size path '{p}' does not match any list column")

    # --- inference scan (only if something is still unknown) -----------------
    stats: Dict[str, dict] = {}
    unknown = [p for p in candidates if p not in explicit]
    if unknown and default_size is None:
        eprint(
            f"Scanning '{args.input}' to infer list lengths for: {', '.join(unknown)}")
        scanned = 0
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=selected):
            for i, f in enumerate(batch.schema):
                collect_stats(batch.column(i), f.name, stats)
            scanned += batch.num_rows
            if args.infer_rows and scanned >= args.infer_rows:
                break
        eprint(f"  scanned {scanned:,} rows")

    # --- resolve sizes -------------------------------------------------------
    sizes: Dict[str, int] = {}
    skipped: List[str] = []
    for p in candidates:
        if p in explicit:
            sizes[p] = explicit[p]
            eprint(f"  {p}: {explicit[p]} (specified)")
            continue
        st = stats.get(p)
        if st and st["n"] > 0 and st["min"] == st["max"]:
            sizes[p] = st["min"]
            eprint(f"  {p}: {st['min']} (inferred from {st['n']:,} lists)")
            continue
        if default_size is not None:
            sizes[p] = default_size
            eprint(f"  {p}: {default_size} (default)")
            continue
        reason = ("no non-null lists" if not st or st["n"] == 0
                  else f"ragged lengths {st['min']}..{st['max']}")
        if args.skip_ragged:
            eprint(f"  {p}: SKIPPED ({reason})")
            skipped.append(p)
        else:
            eprint(f"error: cannot convert '{p}': {reason}. "
                   f"Use --size {p}=N or --skip-ragged.")
            return 1

    if args.inspect:
        out_s = output_schema(in_schema, sizes, set(selected), not args.drop_metadata)
        print("\nResulting schema:\n")
        print(out_s.to_string(show_field_metadata=False))
        return 0

    if not sizes:
        eprint("Nothing to convert.")
        return 1

    # --- rewrite -------------------------------------------------------------
    out_schema = output_schema(in_schema, sizes, set(selected), not args.drop_metadata)
    rg_size = args.row_group_size or args.batch_size

    writer_kwargs = dict(compression=args.compression)
    if args.compression_level is not None:
        writer_kwargs["compression_level"] = args.compression_level

    rows = 0
    pending: List[pa.Table] = []
    pending_rows = 0
    with pq.ParquetWriter(args.output, out_schema, **writer_kwargs) as writer:
        def flush():
            nonlocal pending, pending_rows
            if pending:
                writer.write_table(pa.concat_tables(pending))
                pending, pending_rows = [], 0

        for batch in pf.iter_batches(batch_size=args.batch_size):
            arrays = convert_batch(batch, sizes, set(selected))
            tbl = pa.Table.from_arrays(arrays, schema=out_schema)
            pending.append(tbl)
            pending_rows += tbl.num_rows
            rows += tbl.num_rows
            if pending_rows >= rg_size:
                flush()
            eprint(f"\r  wrote {rows:,} rows", end="")
        flush()
    eprint(f"\rDone: {rows:,} rows -> {args.output}")

    for f in pq.ParquetFile(args.output).schema_arrow:
        if "fixed_size_list" in str(f.type):
            eprint(f"  {f.name}: {f.type}")
    if skipped:
        eprint(f"  left as variable lists: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
