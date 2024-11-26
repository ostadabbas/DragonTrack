import os
import re

def rename_files(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern (leading zeros followed by .jpg)
        match = re.match(r'0+(\d+\.jpg)', filename)
        if match:
            # Extract the number without leading zeros
            number = match.group(1)
            # Define the new filename
            new_filename = f'frame{number}'
            # Full path for old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed "{old_file}" to "{new_file}"')

# Specify the directory where your files are located
directory = '/data/'

# Call the function to rename the files
rename_files(directory)
