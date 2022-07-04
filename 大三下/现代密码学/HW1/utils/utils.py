N = 26

def ch2num(ch: str) -> int:
    return ord(ch) - ord('A')

def num2ch(num: int) -> str:
    return chr(ord('A') + num)

def add(x: int, y: int) -> int:
    return (x + y) % N

def sub(x: int, y: int) -> int:
    return (x - y + N) % N

def calc_freq(cipher):
    freq = {
        num2ch(_): 0 for _ in range(26)
    }
    for ch in cipher:
        freq[ch] += 1
    return freq

def calc_freq_norm(cipher):
    freq = calc_freq(cipher)
    for key, value in freq.items():
        freq[key] = value / len(cipher)
    return freq

def calc_k_letter_freq(cipher, k):
    now = {}
    for i in range(len(cipher) - k):
        if cipher[i : i + k] not in now:
            now[cipher[i : i + k]] = 1
        else:
            now[cipher[i : i + k]] += 1
    ret = now.copy()
    for key, value in now.items():
        if value < 2:
            del ret[key]
    return ret

def calc_word_freq(cipher):
    freq = {}
    for n in range(2, 11):
        update = calc_k_letter_freq(cipher, n)
        freq.update(update)
    
    ret = {}
    for subword in freq:
        flag = True
        for word in freq:
            if word == subword:
                continue
            if subword in word and freq[subword] == freq[word]:
                flag = False
                break
        if flag:
            ret[subword] = freq[subword]
    return ret

def calc_coin(cipher):
    freq = calc_freq(cipher)
    cnt_all = len(cipher)
    res = sum([ freq[num2ch(_)] * (freq[num2ch(_)] - 1) for _ in range(N)]) / cnt_all / (cnt_all - 1)
    return res

def calc_coin_with_length(length, cipher):
    ret = []
    for i in range(length):
        res = calc_coin(cipher[i::length])
        ret.append(str(res))
    return ret

def vigenere_search(cipher, freq_list, length, top):
    for i in range(top ** length):
        shifts_pick = []
        order = i
        for index in range(length):
            let_pick = freq_list[index][order % top][1]
            shifts_pick.append((ch2num('E') - ch2num(let_pick) + N) % N)
            order //= top
        
        text = ''
        for j in range(len(cipher)):
            k = (ch2num(cipher[j]) + shifts_pick[j % length]) % N
            text = text + num2ch(k)
        coin = calc_coin(text)
        if coin >= 0.06:
            print(f'coincidence = {coin}\ntext = {text}')
            judge = input()
            if judge == 'Y' or judge == 'y' or judge == 'yes' or judge == 'Yes' or judge == 'YES':
                return text

def top_freq(cipher, length, top):
    freq_list = []
    for i in range(length):
        freq = calc_freq(cipher[i::length])
        freq = [ (freq[num2ch(_)], num2ch(_)) for _ in range(N) ]
        freq.sort(reverse=True)
        freq_list.append(freq[:top])
    return freq_list