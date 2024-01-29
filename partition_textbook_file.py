import os

input_txtbook_file=r"/Users/arvin/Downloads/textbook_txt_files/2007_Book_FundamentalsOfRenalPathology.txt"
output_dir = "outputs"
output_txtbook_basename=os.path.basename(input_txtbook_file)[:-4]


def split_file_by_paragraphs(input_txtbook_file, output_txtbook_basename, output_dir, num_paragraphs_per_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(input_txtbook_file, 'r') as file:
        paragraphs = file.read().strip().split('\n\n')

    file_number = 1
    paragraph_number = 1
    for i in range(0, len(paragraphs), num_paragraphs_per_file):
        with open(os.path.join(output_dir, f'{output_txtbook_basename}_{file_number}.txt'), 'w') as file:
            for paragraph in paragraphs[i:i + num_paragraphs_per_file]:
                if paragraph.strip():  # Check if paragraph is not just whitespace
                    file.write(str(paragraph_number) + ". " + paragraph + "\n")
                    paragraph_number += 1
        file_number += 1

# Example usage
num_paragraphs_per_file= 50
split_file_by_paragraphs(input_txtbook_file, output_txtbook_basename, output_dir, num_paragraphs_per_file)