def phn_to_textgrid(phn_file, textgrid_file, sampling_rate=16000):
    """
    Convert a TIMIT .PHN file to a Praat TextGrid file.
    
    Args:
    - phn_file: Path to the .PHN file
    - textgrid_file: Path where the .TextGrid file will be saved
    - sampling_rate: Sampling rate of the audio files (default: 16000 for TIMIT)
    """
    with open(phn_file, 'r') as f:
        lines = f.readlines()
    
    # Convert start and end times from samples to seconds
    annotations = [(int(start) / sampling_rate, int(end) / sampling_rate, phoneme)
                   for start, end, phoneme in (line.strip().split() for line in lines)]
    
    with open(textgrid_file, 'w') as f:
        # Write the header
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write('xmin = 0\n')
        xmax = max(annotations, key=lambda x: x[1])[1]  # End time of the last phoneme
        f.write(f'xmax = {xmax}\n')
        f.write('tiers? <exists>\n')
        f.write('size = 1\n')
        f.write('item []:\n')
        f.write('    item [1]:\n')
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "phoneme"\n')
        f.write(f'        xmin = 0\n')
        f.write(f'        xmax = {xmax}\n')
        f.write(f'        intervals: size = {len(annotations)}\n')
        for i, (start, end, phoneme) in enumerate(annotations, start=1):
            f.write(f'        intervals [{i}]:\n')
            f.write(f'            xmin = {start}\n')
            f.write(f'            xmax = {end}\n')
            f.write(f'            text = "{phoneme}"\n')

# Example usage:
phn_file = '/home/andybi7676/Desktop/reborn-uasr/data/text/timit/FCAU0/SX227.PHN'
textgrid_file = '/home/andybi7676/Desktop/reborn-uasr/data/text/timit/FCAU0_SX227.TextGrid'
phn_to_textgrid(phn_file, textgrid_file)