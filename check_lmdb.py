import lmdb
import os

# Path to your LMDB directory
lmdb_path = "lists_info/metadata.lmdb"

# Open the environment in read-only mode
env = lmdb.open(lmdb_path, readonly=True, lock=False)

try:
    # Get environment info and stats
    info = env.info()
    stat = env.stat()
    
    # Print all available keys for debugging
    print("Available info keys:", list(info.keys()))
    print("Available stat keys:", list(stat.keys()))
    
    # Calculate map size
    map_size_gb = info.get('map_size', 0) / (1024**3)
    
    # Get database size - try different approaches
    if 'last_pgno' in info and 'psize' in info:
        data_size_gb = info['last_pgno'] * info['psize'] / (1024**3)
    else:
        # Alternative calculation if standard keys aren't available
        data_size_bytes = os.path.getsize(os.path.join(lmdb_path, "data.mdb"))
        data_size_gb = data_size_bytes / (1024**3)
    
    # Print statistics
    print(f"\nLMDB Statistics for {lmdb_path}:")
    print(f"Map Size (reserved): {map_size_gb:.2f} GB")
    print(f"Actual Data Size: {data_size_gb:.2f} GB")
    print(f"Total Records: {stat.get('entries', 'unknown')}")
    
    # Count records manually if needed
    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0
        for _ in cursor:
            count += 1
        print(f"Counted Records: {count}")
    
    # Check metadata DB specifically
    try:
        metadata_db = env.open_db(b'metadata', txn=env.begin())
        with env.begin() as txn:
            cursor = txn.cursor(db=metadata_db)
            meta_count = 0
            for _ in cursor:
                meta_count += 1
            print(f"Metadata DB Records: {meta_count}")
    except:
        print("Could not access metadata DB specifically")
    
finally:
    env.close()