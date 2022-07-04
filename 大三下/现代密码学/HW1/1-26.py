def encode(cipher, m, n):
    if len(cipher) % (m * n) != 0:
        return False, ""
    text = ""
    for i in range(0, len(cipher), n * m):
        a = cipher[i : i + n * m]
        a = [a[i : i + m] for i in range(0, n * m, m)]
        text += ''.join([''.join([a[i][j] for i in range(n)]) for j in range(m)])
    return True, text

def decode(cipher, m, n):
    return encode(cipher, n, m)

def find(cipher):
    cnt = 1
    while True:
        cnt += 1
        for i in range(1, cnt):
            flag, text = decode(cipher, i, cnt - i)
            if flag == False:
                continue
            judge = input(text + '\n')
            if judge == 'Y' or judge == 'y' or judge == 'yes' or judge == 'Yes' or judge == 'YES':
                print(f'm = {i}, n = {cnt - i}')
                return text

print(decode(encode('cryptography', 4, 3)[1], 4, 3)[1])

cipher = 'myamraruyiqtenctorahroywdsoyeouarrgdernogw'
text = find(cipher)
