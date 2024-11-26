import os
import pickle
import argparse

def main(path_to_pickles, your_file):
    root_path = path_to_pickles
    all_data = []

    # Loop through the files in the directory
    for file in os.listdir(root_path):
        if file.endswith(".pkl"):
            file_path = os.path.join(root_path, file)
            # Load data from the pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                all_data += data

    # Create the mmcv_data dictionary
    mmcv_data = {
        "infos": all_data,
        'metadata': {'version': '1.0-trainval'},
    }

    # Save the mmcv_data to the specified output file
    with open(your_file, "wb") as f:
        pickle.dump(mmcv_data, f)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process pickle files into mmcv format")
    parser.add_argument("path_to_pickles", help="Path to the directory containing the .pkl files")
    parser.add_argument("your_file", help="Output file to save the mmcv_data")

    args = parser.parse_args()

    # Call the main function with the arguments
    main(args.path_to_pickles, args.your_file)
