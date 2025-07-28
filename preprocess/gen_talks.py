import os
import re
import pandas as pd
from config import *
import csv
import datetime
import util as ut

def extract_talk_environment(tex_content: str) -> str:
    """
    Extracts the \begin{talk}...\end{talk} environment from a .tex file content.
    Raises ValueError if no talk environment is found in content.
    """
    pattern = re.compile(r"\\begin\{talk\}.*?\\end\{talk\}", re.DOTALL)
    match = pattern.search(tex_content)
    if not match:
        raise ValueError("No talk environment found")
    return match.group(0)


def extract_session_environment(tex_content: str) -> str:
    """
    Extracts the first \begin{session} ... \end{session} block
    (including those lines) from a .tex file content.
    """
    pattern = re.compile(
        r"^\s*\\begin\{session\}.*?^\s*\\end\{session\}",
        re.DOTALL | re.MULTILINE
    )
    match = pattern.search(tex_content)
    if not match:
        return None
    return match.group(0)

def process_session(
    id_val: str,
    prefix: str,
    tex_dir: str,
    session_time: str,
    session_id: str
) -> str | None:
    """
    Reads <prefix><id_val>.tex in tex_dir, extracts its session environment,
    parses organizer info from \organizer{...}{...}{...} blocks,
    and rebuilds a \begin{session} block with all fields filled properly.
    """
    full_id = f"{prefix}{id_val}"
    tex_path = os.path.join(tex_dir, f"{full_id}.tex")
    if not os.path.isfile(tex_path):
        #print(f"WARN: File not found: {tex_path}")
        return None

    # Read with fallback encoding
    try:
        content = open(tex_path, 'r', encoding='utf-8').read()
    except UnicodeDecodeError:
        content = open(tex_path, 'r', encoding='latin-1').read()

    # Extract the session environment
    block = extract_session_environment(content)
    if block is None:
        print(f"WARN: No session environment found in {tex_path}")
        return None

    # Split into lines and drop \begin and \end tags
    lines = block.splitlines()
    if lines and lines[0].strip().startswith(r'\begin{session}'):
        lines.pop(0)
    if lines and lines[-1].strip().startswith(r'\end{session}'):
        lines.pop()

    # Find session title
    session_title = ""
    title_idx = None
    for idx, line in enumerate(lines):
        m = re.match(r"\s*\{(.+?)\}.*\[1\]", line)
        if m:
            session_title = m.group(1).strip()
            title_idx = idx
            break
    if title_idx is None:
        print(f"WARN: Session title not found in {full_id}")
        return None

    # Move past title line
    i = title_idx + 1
    # Skip number-of-organizers line if present and extract the number to M
    M = 3  # Default number of organizers
    if i < len(lines):
        match = re.match(r"\s*\{(\d+)\}", lines[i])
        if match:
            M = int(match.group(1))
            i += 1

    # Join the rest for organizer matching
    session_body = "\n".join(lines[i:])

    # Find all organizer blocks
    organizer_pattern = re.compile(
        r"""
        \\organizer     # literal “\organizer”
        \s*\{\s*        # open brace for name
        ([^}]*)         # capture everything up to the next }
        \}\s*           # close name‐brace
        [^{}]*          # skip any text (comments, newlines, etc.) up to
        \{\s*           # open brace for affiliation
        ([^}]*)         # capture affiliation
        \}\s*           # close affiliation‐brace
        [^{}]*          # skip up to
        \{\s*           # open brace for email
        ([^}]*)         # capture email
        \}              # close email‐brace
        """,
        re.VERBOSE | re.DOTALL
    )
    organizers = organizer_pattern.findall(session_body)

    # Remove organizer blocks and strip comments
    #print(f"({session_body}")
    session_body_clean = organizer_pattern.sub("", session_body)

    # remove lines  "{}% organizer one email" from session_body_clean
    session_body_clean = re.sub(r"\s*\{\}\s*%\s*organizer one email", "", session_body_clean)
    # remove lines  "{}% organizer two email" from session_body_clean
    session_body_clean = re.sub(r"\s*\{\}\s*%\s*organizer two email", "", session_body_clean)
    # remove lines  "{}% organizer three email" from session_body_clean
    session_body_clean = re.sub(r"\s*\{\}\s*%\s*organizer three email", "", session_body_clean)

    body_lines = []
    for line in session_body_clean.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        # Skip any stray end tag
        if stripped.startswith(r'\end{session}'):
            continue

        # Define patterns to remove specific organizer lines
        remove_patterns = [
            r"^\s*\{\}\s*\%\s*(organizer|orgnizer)\s*.*$",
        ]
        
        # Skip line if it matches any of the removal patterns
        if any(re.match(pattern, stripped) for pattern in remove_patterns):
            continue

        # Ensure lines are properly cleaned
        body_lines.append(stripped.strip())

    #print(f"{session_title = }, {M = }, {organizers = }, \n{session_body_clean = }")

    # Compose the new session block
    output_lines = []
    output_lines.append(r"\begin{session}")
    output_lines.append(f" {{{session_title}}}% [1] session title")
    # Provide default empty organizers if not enough found
    safe_organizers = organizers + [("", "", "")] * (3 - len(organizers))
    output_lines.append(f" {{{safe_organizers[0][0].strip()}}}% [2] organizer one name")
    output_lines.append(f" {{{safe_organizers[0][1].strip()}}}% [3] organizer one affiliations")
    output_lines.append(f" {{{safe_organizers[0][2].strip()}}}% [4] organizer one email")
    if M > 1:
        output_lines.append(f" {{{safe_organizers[1][0].strip()}}}% [5] organizer two name. Leave unchanged if there is no second organizer, otherwise fill in accordingly.")
        output_lines.append(f" {{{safe_organizers[1][1].strip()}}}% [6] affiliations. Leave unchanged if there is no second organizer, otherwise fill in accordingly.")
        output_lines.append(f" {{{safe_organizers[1][2].strip()}}}% [7] email. Leave unchanged if there is no second organizer, otherwise fill in accordingly.")
    else:
        output_lines.append("{}{}{}")
    output_lines.append(f" {{{session_id}}}% [8] session id")
    if M > 2:
        output_lines.append(
            f" {{\\thirdorganizer{{{safe_organizers[2][0].strip()}}}{{{safe_organizers[2][1].strip()}}}{{{safe_organizers[2][2].strip()}}}}}% [9] third organizer, if any"
        )
    else:
        output_lines.append("{}")

    # Append any remaining content (e.g., abstracts or notes)
    if body_lines:
        output_lines.append("")  # blank line before body
        for bl in body_lines:
            output_lines.append(f" {bl}")
            
    output_lines.append('\end{session}')
    output_lines.append('')
    output_lines.append(f"\\input{{sess{session_id}.tex}}") 
    output_lines.append('')
    output_lines.append("\\clearpage")
    output_lines.append('')

    return "\n".join(output_lines)


def load_ids(csv_path: str) -> list[str]:
    """
    Reads CSV and returns a cleaned list of IDs from TalkID or SessionID column.
    """
    df = pd.read_csv(csv_path, dtype=str)
    for col in ('TalkID', 'SessionID'):
        if col in df.columns:
            series = df[col].dropna().astype(str).str.strip()
            return [val for val in series if val and val.lower() != 'nan']
    raise KeyError("CSV must contain 'TalkID' or 'SessionID'.")


def sort_ids(ids: list[str]) -> list[str]:
    """
    Performs a natural sort on IDs, handling numeric and hyphen components.
    """
    try:
        return sorted(
            ids,
            key=lambda s: [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', s)]
        )
    except Exception:
        return ids


def format_full_id(id_val: str, prefix: str) -> str:
    """
    Prepends prefix to id_val if not already present.
    """
    return f"{prefix}{id_val}" if prefix and not id_val.startswith(prefix) else id_val

def process_talk(id_val: str, prefix: str, tex_dir: str, session_time:str, session_id:str) -> str | None:
    """
    Extracts and normalizes a talk environment from a .tex file.
    Ensures exactly 9 argument slots, fills missing with "",
    overrides talk-id ([8]) with full_id, sets field 9 to "photo",
    preserves the abstract body, and optionally adds \clearpage for plenary.
    """
    full_id = format_full_id(id_val, prefix)
    tex_path = os.path.join(tex_dir, f"{full_id}.tex")
    if not os.path.isfile(tex_path):
        #print(f"WARN: File not found: {tex_path}")
        return None

    # Read file with fallback encoding
    try:
        content = open(tex_path, 'r', encoding='utf-8').read()
    except UnicodeDecodeError:
        content = open(tex_path, 'r', encoding='latin-1').read()

    # Extract the talk block
    try:
        block = extract_talk_environment(content)
    except ValueError:
        print(f"WARN: talk cannot be extracted: {tex_path}")
        return None

    # Split into lines for body extraction
    raw_lines = block.splitlines()
    # Locate begin/end
    try:  # \being{talk} and \end{talk} should not be preceded with any spaces
        begin_idx = next(i for i, l in enumerate(raw_lines) if re.match(r"^\\begin\{talk\}", l))
        end_idx   = next(i for i, l in enumerate(raw_lines) if re.match(r"^\\end\{talk\}", l))
    except StopIteration:
        return None
    
    # --- Identify field lines (slots 1–N, N=9 for plenary talks and contributed talks)
    N = 9
    # compile a single regex that both identifies a field-line
    # and captures everything between the {…} (including any inner braces)
    field_re = re.compile(r"""
        ^\s*          # optional leading space
        \{(.+?)\}     # capture ALL text inside the outer {…}, non-greedy
        \s*%\s*       # then some space and a literal %
        \[\d+\]       # then [slot-number]
        """, re.VERBOSE)

    # find field indices
    field_idxs = [i for i, l in enumerate(raw_lines) if field_re.match(l)]
    last_field_idx = max(field_idxs) if field_idxs else begin_idx

    # extract body
    body_lines = raw_lines[last_field_idx+1 : end_idx]
    

    # parse out the slots
    raw_args = [ field_re.match(l).group(1)
                for l in raw_lines
                if field_re.match(l) ]

    # Initialize slots 1–N to empty
    args = {i: '' for i in range(1, N+1)}
    # Fill from parsed values, in order
    for idx, val in enumerate(raw_args, start=1):
        if idx > N:
            break
        args[idx] = val.strip()

    # Override specific slots
    args[7] = session_time
    args[8] = full_id  # talk id
    args[9] = 'photo' if prefix=="P" else session_id  

    # Build normalized talk block with descriptions
    descriptions = {
        1: 'talk title',
        2: 'speaker name',
        3: 'affiliations',
        4: 'email',
        5: 'coauthors',
        6: 'special session',
        7: 'time slot',
        8: 'talk id',
        9: 'session id or photo',
    }

    lines = ['\\begin{talk}']
    for i in range(1, N+1):
        lines.append(f"  {{{args[i]}}}% [{i}] {descriptions[i]}")

    # Append existing abstract body, preserving indent
    for bl in body_lines:
        lines.append(bl)
    lines.append('\\end{talk}')
    lines.append('')

    # For plenary talks (prefix 'P'), add a page break after
    if prefix == 'P':
        lines.append('\\clearpage')

    return '\n'.join(lines)


def write_output(blocks: list[str], output_path: str, chapter: str = 'Plenary Talks') -> None:
    """
    Writes header and talk blocks to the output .tex file.
    """
    # Modify header based on document type
    if chapter == "Plenary Talks":
        header = f"\\chapter{{{chapter}}}\\newpage\n\n"
    elif chapter == "Special Sessions":
        header = f"\\chapter{{{chapter}}}\n\n"
    elif chapter == "Special Session Talks":
        header = f"\\chapter{{Abstracts}}\\newpage\\section{{{chapter}}}\n\n"
    else:
        header = f"\\section{{{chapter}}}\n\n"
        
    body = "\n".join(blocks)
    # remove only the last '\clearpage' (plus any trailing backslash/newline)
    body = re.sub(
        r"(?s)(.*)\\clearpage\s*\\?$",
        r"\1",
        body
    )

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(header)
        out.write(body)

    print(f"Output: {output_path}")

def generate_tex_talks(csv_path: str = "plenary_abstracts_talkid.csv",
                       tex_dir: str = ".",
                       output_path: str = "plenary_talks.tex",
                       strict: bool = False,
                       prefix: str = ""):
    """
    Main entry: load IDs, sort them, process each talk, and write output.
    """
    # Read full table so we can build an ID → SessionTime lookup
    df = pd.read_csv(csv_path, dtype=str)
    id_col = "TalkID" if "TalkID" in df.columns else "SessionID"
    session_time_map = dict(
        zip(
            df[id_col].astype(str).str.strip(),
            df.get("SessionTime", pd.Series()).fillna("").astype(str),
        )
    )
    talk_time_map = dict(
        zip(
            df[id_col].astype(str).str.strip(),
            df.get("EventTime", pd.Series()).fillna("").astype(str),
        )
    )
    id_map = dict(
        zip(
            df[id_col].astype(str).str.strip(),
            df.get("SessionID", pd.Series()).fillna("").astype(str),
        )
    )

    # Load and sort IDs 
    ids = load_ids(csv_path)
    sorted_ids = sort_ids(ids)

    blocks = []
    missing = []

    for id_val in sorted_ids:
        session_time = session_time_map.get(id_val, "")
        talk_time = talk_time_map.get(id_val, "")
        session_id = id_map.get(id_val, "")
        if prefix == "":
            block = process_session(id_val, prefix, tex_dir, session_time, session_id)
        else:
            block = process_talk(id_val, prefix, tex_dir, talk_time, session_id)
        if block is None:
            missing.append(format_full_id(id_val, prefix))
            # Create dummy abstract .tex files in abstracts directory
            dummy_tex_path = os.path.join(tex_dir, f"{format_full_id(id_val, prefix)}.tex")
            with open(dummy_tex_path, 'w', encoding='utf-8') as f:
                # Try to get as much info as possible from the CSV
                speaker = df.loc[df[id_col] == id_val, "Speaker"].values[0] if "Speaker" in df.columns and any(df[id_col] == id_val) else "TBD"
                title = df.loc[df[id_col] == id_val, "Title"].values[0] if "Title" in df.columns and any(df[id_col] == id_val) else "TBD"
                affiliations = df.loc[df[id_col] == id_val, "Affiliations"].values[0] if "Affiliations" in df.columns and any(df[id_col] == id_val) else ""
                email = df.loc[df[id_col] == id_val, "Email"].values[0] if "Email" in df.columns and any(df[id_col] == id_val) else ""
                coauthors = df.loc[df[id_col] == id_val, "Coauthors"].values[0] if "Coauthors" in df.columns and any(df[id_col] == id_val) else ""
                special_session = df.loc[df[id_col] == id_val, "SpecialSession"].values[0] if "SpecialSession" in df.columns and any(df[id_col] == id_val) else ""
                time_slot = talk_time if talk_time else session_time
                session_id_val = session_id if session_id else ""
                photo_or_session = session_id_val
                f.write(
                    "\\begin{talk}\n"
                    f"  {{{title}}}% [1] talk title\n"
                    f"  {{{speaker}}}% [2] speaker name\n"
                    f"  {{{affiliations}}}% [3] affiliations\n"
                    f"  {{{email}}}% [4] email\n"
                    f"  {{{coauthors}}}% [5] coauthors\n"
                    f"  {{{special_session}}}% [6] special session. Leave this field empty for contributed talks. \n"
                    "                % Insert the title of the special session if you were invited to give a talk in a special session.\n"
                    f"  {{{time_slot}}}% [7] time slot\n"
                    f"  {{{format_full_id(id_val, prefix)}}}% [8] talk id\n"
                    f"  {{{photo_or_session}}}% [9] session id or photo\n"
                    "\\end{talk}\n"
                )
        else:
            blocks.append(block)

    # If strict mode, error out on any missing talks
    if strict and missing:
        raise RuntimeError(f"Missing talks for IDs: {', '.join(missing)}")

    chapter = (
        "Plenary Talks" if prefix == "P"
        else "Contributed Talks" if prefix == "T"
        else "Special Session Talks" if prefix == "S"
        else "Special Sessions"
    )

    # Final clean up of blocks using util function:
    blocks = [ut.clean_tex_content(block) for block in blocks]

    write_output(blocks, output_path, chapter)

    # Warn if any were missing
    if missing:
        print(f"WARN: Missing talk abstracts for IDs: {missing}")


def generate_individual_session_files():
    """
    Generate individual .tex files for each session (special sessions and technical sessions).
    For each session ID (e.g., S1, T4), reads all session-*.tex files and outputs
    the content to preprocess/out/SessionID_sess_talks.tex
    """
    import glob
    
    # Process both special sessions and technical sessions
    session_types = [
        {
            'csv_path': os.path.join(interimdir, "special_session_submissions_sessionid.csv"),
            'session_type': 'Special Sessions'
        },
        {
            'csv_path': os.path.join(interimdir, "contributed_talk_submissions_talkid.csv"),
            'session_type': 'Technical Sessions'
        }
    ]
    
    abstracts_dir = os.path.join(indir, 'abstracts')
    
    for session_config in session_types:
        csv_path = session_config['csv_path']
        session_type = session_config['session_type']
        
        if not os.path.exists(csv_path):
            print(f"WARN: CSV file not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path, dtype=str)
        if 'SessionID' not in df.columns:
            print(f"WARN: SessionID column not found in {csv_path}")
            continue
        
        # Get unique session IDs  
        session_ids = df['SessionID'].dropna().astype(str).str.strip().unique()
        session_ids = [sid for sid in session_ids if sid and sid.lower() != 'nan']
        
        print(f"Processing {session_type}: {len(session_ids)} sessions")
        
        for session_id in session_ids:
            # Find all abstract files for this session (e.g., S10-1.tex, T4-1.tex, etc.)
            pattern = os.path.join(abstracts_dir, f"{session_id}-*.tex")
            abstract_files = glob.glob(pattern)
            
            if not abstract_files:
                print(f"WARN: No abstract files found for session {session_id}")
                continue
            
            # Sort files naturally (S10-1.tex, S10-2.tex, T4-1.tex, T4-2.tex, etc.)
            abstract_files = sorted(abstract_files, key=lambda x: [
                int(tok) if tok.isdigit() else tok 
                for tok in re.split(r'(\d+)', os.path.basename(x))
            ])
            
            # Process each abstract file and collect the talk blocks
            talk_blocks = []
            for abs_file in abstract_files:
                try:
                    # Read file with fallback encoding
                    try:
                        content = open(abs_file, 'r', encoding='utf-8').read()
                    except UnicodeDecodeError:
                        content = open(abs_file, 'r', encoding='latin-1').read()
                    
                    # Extract talk environment
                    talk_block = extract_talk_environment(content)
                    
                    # Clean the talk block using util function
                    talk_block = ut.clean_tex_content(talk_block)
                    talk_blocks.append(talk_block)
                    
                except Exception as e:
                    print(f"WARN: Error processing {abs_file}: {e}")
                    continue
            
            if not talk_blocks:
                print(f"WARN: No valid talk blocks found for session {session_id}")
                continue
            
            # Write output file
            output_path = os.path.join(outdir, f"{session_id}_sess_talks.tex")
            
            # Get session info for header
            session_info = df[df['SessionID'] == session_id].iloc[0] if len(df[df['SessionID'] == session_id]) > 0 else None
            session_title = session_info['SessionTitle'] if session_info is not None and 'SessionTitle' in session_info else f"Session {session_id}"
            
            # Write header and content
            with open(output_path, 'w', encoding='utf-8') as out:
                out.write(f"\\section{{{session_title}}}\\newpage\n\n")
                out.write("\n\n".join(talk_blocks))
                out.write("\n")
            
            print(f"Generated: {output_path} (with {len(talk_blocks)} talks)")


if __name__ == '__main__':
    # map sheet keys to filename prefixes
    prefix_map = {
        'plenary_abstracts': 'P',
        'special_session_submissions': '',
        'special_session_abstracts': 'S',
        'contributed_talk_submissions': 'T'
    }

    for key in ["special_session_submissions", "plenary_abstracts", "special_session_abstracts", "contributed_talk_submissions"]:  
        if key in prefix_map:
            if key == "special_session_submissions":
                csv_path = os.path.join(interimdir, f"{key}_sessionid.csv")
            else:
                csv_path = os.path.join(interimdir, f"{key}_talkid.csv")
            tex_dir = os.path.join(indir, 'abstracts')
            out_path = os.path.join(outdir, f"{key}_talks.tex")
            generate_tex_talks(
                csv_path=csv_path,
                tex_dir=tex_dir,
                output_path=out_path,
                strict=False,
                prefix=prefix_map[key]
            )
    
    # Generate individual session files for special and technical sessions
    generate_individual_session_files() 
    