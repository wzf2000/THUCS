from .models import BinaryModel, TernaryModel
import sys
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default = 'TernaryModel', help = 'The model name.', choices = ['BinaryModel', 'TernaryModel'])
    parser.add_argument('--input', '-i', default = 'data/input.txt', help = 'The input file.')
    parser.add_argument('--output', '-o', default = 'data/test.txt', help = 'The output file.')
    parser.add_argument('--nocheck', '-n', default = 0, nargs = '?', const  = 1, type = bool)
    parser.add_argument('--answer', '-a', default = 'data/output.txt', help = 'The answer file.')
    args = parser.parse_args()
    
    model_name = eval(args.model)
    model = model_name()
    input_path = args.input
    output_path = args.output
    val_path = args.answer
    with open(output_path, 'w', encoding = 'utf8') as output:
        with open(input_path, 'r', encoding = 'utf8') as input:
            sentences = input.readlines()
        output_sentences = []
        for sentence in tqdm(sentences):
            res = model.forward(sentence.strip())
            output.write(res + '\n')
            output_sentences.append(res)
        if not args.nocheck:
            try:
                with open(val_path, 'r', encoding = 'utf8') as val:
                    val_sentences = val.readlines()
                cnt = 0
                sum = 0
                per_sum = 0
                for i in tqdm(range(len(sentences))):
                    if output_sentences[i] == val_sentences[i].strip():
                        cnt += 1
                    for j in range(len(output_sentences[i])):
                        if output_sentences[i][j] == val_sentences[i][j]:
                            per_sum += 1
                    sum += len(output_sentences[i])
                print('整句正确率：', str(int(cnt / len(sentences) * 10000) / 100) + '%')
                print('逐字正确率：', str(int(per_sum / sum * 10000) / 100) + '%')
            except Exception as e:
                print("Open answer file failed!", e)
            
