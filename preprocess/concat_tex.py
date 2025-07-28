import os
import re
from pathlib import Path

def consolidate_latex_document():
    """
    Consolidate MCM2025_book.tex by replacing all \input{...} commands 
    with the actual content of the referenced files.
    """
    
    # Define paths
    base_dir = Path("/Users/terrya/Documents/ProgramData/MCM-2025-Program")
    mcm_tex_dir = base_dir / "MCM_ProgramBook_TEX"
    output_dir = mcm_tex_dir  #base_dir / "preprocess" / "output"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the main MCM2025_book.tex file
    main_file = mcm_tex_dir / "MCM2025_book.tex"
    
    if not main_file.exists():
        print(f"Error: Main file not found at {main_file}")
        return
    
    print(f"Reading main file: {main_file}")
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Processing \\input{} commands...")
    
    # Function to recursively replace \input{filename} with file content
    def replace_input_recursive(tex_content):
        input_pattern = r'\\input\{([^}]+)\}'
        def replacer(match):
            filename = match.group(1).strip()
            # Remove .tex extension if present, then add it back
            if filename.endswith('.tex'):
                base_filename = filename[:-4]
            else:
                base_filename = filename
            possible_files = [
                mcm_tex_dir / f"{base_filename}.tex",
                mcm_tex_dir / f"{filename}",
                mcm_tex_dir / filename
            ]
            input_file = None
            for pf in possible_files:
                if pf.exists():
                    input_file = pf
                    break
            if input_file is None:
                print(f"Warning: Could not find input file for \\input{{{filename}}}")
                return f"% WARNING: File {filename} not found\n"
            print(f"  -> Including: {input_file.name}")
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                # Recursively process \input{} in the included file
                file_content = replace_input_recursive(file_content)
                header = f"\n% ============================================================================\n"
                header += f"% Content from: {input_file.name}\n"
                header += f"% ============================================================================\n\n"
                return header + file_content + "\n"
            except Exception as e:
                print(f"Error reading {input_file}: {e}")
                return f"% ERROR: Could not read {filename}: {e}\n"
        return re.sub(input_pattern, replacer, tex_content)

    # Recursively replace all \input{...} commands with file content
    consolidated_content = replace_input_recursive(content)
    
    # Write the consolidated file
    output_file = output_dir / "MCM2025_consolidated.tex"
    print(f"\nWriting consolidated file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(consolidated_content)
    
    print(f"\n✅ Successfully created consolidated LaTeX file!")
    print(f"📄 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size:,} bytes")
    
    # Count the number of input files that were processed
    input_pattern = r'\\input\{([^}]+)\}'
    input_count = len(re.findall(input_pattern, content))
    print(f"🔗 Processed {input_count} \\input{{}} commands")
    
    return output_file

def main():
    """Main function to run the consolidation"""
    print("🔄 Starting LaTeX document consolidation...")
    print("=" * 60)
    
    try:
        output_file = consolidate_latex_document()
        
        if output_file and output_file.exists():
            print("\n" + "=" * 60)
            print("✅ Consolidation completed successfully!")
            print(f"📁 Consolidated file created at:")
            print(f"   {output_file}")
            print("\n💡 You can now use this single .tex file instead of the")
            print("   multiple files with \\input{} commands.")
            print("\n🚀 To compile the PDF:")
            print(f"   cd {output_file.parent}")
            print(f"   pdflatex {output_file.name}")
            
        else:
            print("\n❌ Consolidation failed!")
            
    except Exception as e:
        print(f"\n❌ Error during consolidation: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()