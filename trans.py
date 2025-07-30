import os
import subprocess

translator = 'google'
target_lang = 'en'  # All texts will be translated to English

def read_input(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def translate_text(text, translator, target_lang):
    #cmd = ['trans', f'--translators={translator}', '-b', f':{target_lang}', text]
    cmd = ['trans', f'--translators={translator}', f':{target_lang}', text]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"[{translator}] Translate Shell error: {result.stderr.strip()}")

    return result.stdout.strip()

def write_output(filepath, translated_text):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(translated_text)

def process_folder(input_folder, output_folder, limit=None):
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            if limit is not None and count >= limit:
                break

            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_{translator}.txt"
            output_path = os.path.join(output_folder, output_filename)

            print(f"Translating {input_path}...")
            original_text = read_input(input_path)
            translated_text = translate_text(original_text, translator, target_lang)
            write_output(output_path, translated_text)
            print(f"Saved translated file to {output_path}")

            count += 1

def main():
    for lang in ['fas', 'rus', 'zho']:
        input_folder = os.path.join('..', lang)
        output_folder = os.path.join('..', f"{lang}_{translator}")
        process_folder(input_folder, output_folder, limit=3)

if __name__ == '__main__':
    main()
