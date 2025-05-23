import struct
import os

def read_list_sizes(filename="lists_info/sizes.bin"):
    """
    Reads a binary file containing a sequence of uint32_t values.

    Args:
        filename (str): The name of the file to read.

    Returns:
        list[int] or None: A list of integers if successful, None otherwise.
    """
    sizes = []
    uint32_size = 4  # Size of uint32_t in bytes
    
    try:
        with open(filename, 'rb') as f:
            while True:
                chunk = f.read(uint32_size)
                if not chunk:  # End of file
                    break
                if len(chunk) < uint32_size:
                    print(f"Warning: Encountered a partial chunk of size {len(chunk)} at the end of '{filename}'. Expected multiple of {uint32_size}. Ignoring.")
                    break
                # '<I' means little-endian unsigned int (4 bytes)
                size_tuple = struct.unpack('<I', chunk)
                sizes.append(size_tuple[0])
        return sizes
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
        return None

def load_and_verify_ids(list_sizes, ids_filepath="lists_info/ids.bin", sample_lists_to_print=3):
    """
    Loads list IDs from a binary file based on provided list sizes,
    prints a sample of IDs, and verifies file consumption.

    Args:
        list_sizes (list[int]): A list of sizes for each list.
        ids_filepath (str): The path to the IDs file.
        sample_lists_to_print (int): How many non-empty lists to print IDs for.
    """
    print(f"\\nAttempting to load and verify IDs from '{ids_filepath}'...")
    int64_size = 8  # Size of int64_t in bytes
    total_ids_read_from_file = 0
    lists_printed_count = 0

    try:
        with open(ids_filepath, 'rb') as f:
            for i, current_list_size in enumerate(list_sizes):
                if current_list_size > 0:
                    bytes_to_read = current_list_size * int64_size
                    chunk = f.read(bytes_to_read)

                    if len(chunk) < bytes_to_read:
                        print(f"ERROR: List {i}: Expected to read {bytes_to_read} bytes for {current_list_size} IDs, but got only {len(chunk)} bytes (premature EOF).")
                        # It's problematic to continue if an expected read fails midway
                        return 
                    
                    # Unpack the IDs. Format: '<' for little-endian, 'q' for int64_t
                    # Create format string like '<10q' if current_list_size is 10
                    format_string = f'<{current_list_size}q'
                    ids_tuple = struct.unpack(format_string, chunk)
                    total_ids_read_from_file += len(ids_tuple)

                    if lists_printed_count < sample_lists_to_print:
                        print(f"  List {i} (size {current_list_size}): First 10 IDs: {list(ids_tuple[:10])}{'...' if len(ids_tuple) > 10 else ''}")
                        lists_printed_count += 1
                # If current_list_size is 0, we read nothing for this list, which is correct.
            
            # Check if there's any data left in the file
            remaining_data = f.read(1)
            if remaining_data:
                print(f"WARNING: Extra data found at the end of '{ids_filepath}'. Expected EOF.")
            else:
                print(f"Successfully reached EOF for '{ids_filepath}' as expected.")

        print(f"Successfully read a total of {total_ids_read_from_file} IDs from '{ids_filepath}'.")
        
        # Sanity check: sum of list_sizes should equal total_ids_read_from_file
        expected_total_ids = sum(list_sizes)
        if total_ids_read_from_file == expected_total_ids:
            print(f"Verification passed: Total IDs read ({total_ids_read_from_file}) matches sum of list_sizes ({expected_total_ids}).")
        else:
            print(f"ERROR: Verification failed! Total IDs read ({total_ids_read_from_file}) does NOT match sum of list_sizes ({expected_total_ids}).")

    except FileNotFoundError:
        print(f"Error: File '{ids_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred while processing '{ids_filepath}': {e}")

if __name__ == "__main__":
    filepath = "lists_info/sizes.bin"
    list_sizes = read_list_sizes(filepath)

    if list_sizes is not None:
        if list_sizes:
            print(f"Successfully read {len(list_sizes)} list sizes from '{filepath}'.")
            print(f"First 10 sizes: {list_sizes[:10]}")

            print(f"List 167: {list_sizes[167]}")
            
            min_size = min(list_sizes)
            max_size = max(list_sizes)
            # Calculate average only if there are sizes, to avoid division by zero
            avg_size = sum(list_sizes) / len(list_sizes) if len(list_sizes) > 0 else 0.0
            zero_count = list_sizes.count(0)

            print(f"Min list size: {min_size}")
            print(f"Max list size: {max_size}")
            print(f"Average list size: {avg_size:.2f}")
            print(f"Number of empty lists (size 0): {zero_count}")
            print(f"Verify sum of list sizes: {sum(list_sizes)}")
            
            # Verification: File size should be len(list_sizes) * 4
            expected_file_size = len(list_sizes) * 4
            try:
                actual_file_size = os.path.getsize(filepath)
                if actual_file_size == expected_file_size:
                    print(f"File size verification passed: {actual_file_size} bytes.")
                else:
                    print(f"Warning: File size mismatch. Expected {expected_file_size} bytes, got {actual_file_size} bytes.")
            except OSError as e:
                print(f"Could not verify file size: {e}")

            # Now, load and verify the IDs using the read sizes
            load_and_verify_ids(list_sizes, ids_filepath="lists_info/ids.bin")

        else:
            # This case handles if read_list_sizes returned an empty list
            # (e.g. file was empty but readable, or only contained partial chunks ignored)
            print(f"No valid list sizes found in '{filepath}', or the file was empty.")
    else:
        # This case handles if read_list_sizes returned None (e.g. file not found, or other read error)
        print(f"Could not process '{filepath}'.")

# To run this script, ensure 'sizes.bin' is in the same directory
# and then execute: python main.py