def change_one_letter(word, letter, symbol='*'):
    word = list(word)
    for idx, let in enumerate(list(word)):
        if let.lower() == letter.lower():
            word[idx] = symbol

    return ''.join(word)


if __name__ == '__main__':
    print(change_one_letter('Germany', 'A'))



