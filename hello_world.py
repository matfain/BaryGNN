def save_text_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

def main():
    text = "Hello, World!"
    filename = "hello.txt"
    save_text_file(text, filename)

if __name__ == "__main__":
    main()