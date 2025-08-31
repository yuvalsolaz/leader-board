import os

gitignore = ["./venv", "__pycache__", "utils/__pycache__",".idea", "models/__pycache__",
             "data",".gradio", "venv", "notebook", ".vscode"
]

def concatenate_python_files(root_dir, output_file):
    """
    Traverse root_dir, find all .py files, and concatenate their contents
    into output_file with file names as separators.

    Args:
        root_dir (str): The root directory to start the search
        output_file (str): The path to the output text file
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through the directory recursively
        for dirpath, _, filenames in os.walk(root_dir):
            if dirpath.startswith('./venv'):
                continue

            # Filter for .py files
            python_files = [f for f in filenames if f.endswith('.py')]

            for filename in python_files:
                file_path = os.path.join(dirpath, filename)
                # Write the file name as a separator
                outfile.write(f'\n\n===== {file_path} =====\n\n')

                try:
                    # Read and write the file contents
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    # Handle any errors (e.g., permission issues, encoding errors)
                    outfile.write(f'\n[Error reading {file_path}: {str(e)}]\n')


if __name__ == '__main__':
    # Specify the project folder and output file
    project_folder = '.'  # Current directory; change to your project folder path
    output_file = 'all_python_code.txt'

    # Run the concatenation
    concatenate_python_files(project_folder, output_file)
    print(f"All Python code concatenated into {output_file}")