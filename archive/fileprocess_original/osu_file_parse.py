class OSUData:
    def __init__(self):
        self.sections = {
            "General": {},
            "Editor": {},
            "Metadata": {},
            "Difficulty": {},
            "Events": {},
            "TimingPoints": [],
            "HitObjects": [],
        }

    def read_from_file(self, filename):
        current_section = None

        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith("["):
                current_section = line[1:-1]
                continue

            if current_section is None:
                continue

            parts = line.split(":")
            if len(parts) >= 2:
                key = parts[0].strip()
                value = ":".join(parts[1:]).strip()

                if current_section == "TimingPoints":
                    timing_point = [
                        float(v) if "." in v else int(v) for v in value.split(",")
                    ]
                    self.sections["TimingPoints"].append(timing_point)
                elif current_section == "HitObjects":
                    hit_object = [
                        int(v) if "." not in v else float(v) for v in value.split(",")
                    ]
                    self.sections["HitObjects"].append(hit_object)
                else:
                    self.sections[current_section][key] = value

    def save_to_file(self, filename):
        with open(filename, "w") as f:
            for section, data in self.sections.items():
                f.write(f"[{section}]\n")
                if isinstance(data, list) and len(data) > 0:
                    for entry in data:
                        line = ",".join(str(v) for v in entry)
                        f.write(f"{line}\n")
                elif isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"{key}: {value}\n")
                f.write("\n")
