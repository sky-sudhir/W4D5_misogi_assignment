def sanitize_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i == 0:
                # Write header as is
                outfile.write(line)
                continue

            parts = line.rstrip('\n').rsplit(',', 1)  # Split only on the last comma
            if len(parts) == 2:
                transcript, label = parts
                transcript = transcript.strip()
                label = label.strip()

                # Add quotes if transcript contains a comma and is not already quoted
                if ',' in transcript and not (transcript.startswith('"') and transcript.endswith('"')):
                    transcript = f'"{transcript}"'

                outfile.write(f'{transcript},{label}\n')
            else:
                print(f"Skipping malformed line {i+1}: {line.strip()}")

# Example usage
sanitize_csv('sales_data.csv', 'sanitized_output.csv')
