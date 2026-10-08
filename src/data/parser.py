from pathlib import Path
def parse_icbhi_filename(filename):
    """
    Parse an ICBHI respiratory sound filename.

    Expected format:
    patient_recording_location_mode_equipment.wav
    """

    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) != 5:
        raise ValueError(
            f"Unexpected ICBHI filename format: {filename}"
        )

    patient_id = int(parts[0])
    recording_id = parts[1]
    chest_location = parts[2]
    acquisition_mode = parts[3]
    equipment = parts[4]

    return {
        "patient_id": patient_id,
        "recording_id": recording_id,
        "chest_location": chest_location,
        "acquisition_mode": acquisition_mode,
        "equipment": equipment,
    }
    
def parse_icbhi_annotation(annotation_file):
    """
    Parse an ICBHI respiratory cycle annotation file.

    Each row contains:
    start_time    end_time    crackles    wheezes
    """

    cycles = []

    with open(annotation_file, "r") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            parts = line.split()

            if len(parts) != 4:
                raise ValueError(
                    f"Unexpected annotation format in "
                    f"{annotation_file} at line {line_number}: {line}"
                )

            start_time = float(parts[0])
            end_time = float(parts[1])
            crackles = int(parts[2])
            wheezes = int(parts[3])

            cycles.append({
                "start_time": start_time,
                "end_time": end_time,
                "crackles": crackles,
                "wheezes": wheezes,
            })

    return cycles