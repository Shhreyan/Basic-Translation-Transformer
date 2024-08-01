import os
import csv

def load_translation_dict(filename):
    translation_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 2:
                original, translated = row
                translation_dict[original] = translated
    return translation_dict

def translate_text(text, translation_dict):
    translated_text = ''.join(translation_dict.get(char, char) for char in text)
    return translated_text

def main():
    input_filename = 'input.txt'
    output_filename = 'translated.txt'
    translation_filename = 'translation.csv'
    
    # Print the current working directory and list files
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in the directory: {os.listdir()}")

    # Check if the input file exists in the same directory
    if not os.path.isfile(input_filename):
        print(f"Error: {input_filename} not found in the current directory.")
        return

    # Load the translation dictionary from CSV
    if not os.path.isfile(translation_filename):
        print(f"Error: {translation_filename} not found in the current directory.")
        return
    
    translation_dict = load_translation_dict(translation_filename)

    # Read the input file
    with open(input_filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Translate the text
    translated_text = translate_text(text, translation_dict)

    # Write the translated text to the output file
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(translated_text)
    
    print(f"Translation complete. Translated text saved to {output_filename}")

if __name__ == '__main__':
    main()
