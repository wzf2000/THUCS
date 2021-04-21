
with open('input_output_test.txt', 'r', encoding = 'utf8') as f:
    lines = f.readlines()
l = len(lines)
f1 = open('input.txt', 'w', encoding = 'utf8')
f2 = open('output.txt', 'w', encoding = 'utf8')
for i in range(l):
    if i % 2 == 0:
        f1.write(lines[i])
    else:
        f2.write(lines[i])