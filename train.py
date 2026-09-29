from src.cli import Parser, Application


def main():
    return Application().run(Parser.parse())


if __name__ == "__main__":
    main()