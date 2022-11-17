import argparse

class CommandParser:
  def __init__(self) -> None:
    self.parser = argparse.ArgumentParser(prog="anti_cursing", description="anti_cursing")
    self.subparser = self.parser.add_subparsers(
        dest="command",
        help="this is prototype of anti_cursing. there's nothing you have to manipulate"
    )

  def get_args(self) -> argparse.Namespace:
    insutance_parser = self.subparser.add_parser(
        "anti_cursing",
        help="the package of detecting and filtering curse words by using deep learning",
    )
    options_group = insutance_parser.add_mutually_exclusive_group(required=True)
    options_group.add_argument("--name", action="store_true", help=f"name of the package")
    options_group.add_argument("--env", action="store_true", help=f"Environment")

    return self.parser.parse_args()

def main():
  parser = CommandParser()
  args = parser.get_args()

  if args.command == "anti_cursing":
    if args.name:
      print("ANTI-CURSING")
    elif args.env:
      print("python3.8")

if __name__=="__main__":
  parser = CommandParser()
  args = parser.get_args()
  print(args)
  print(args.my_name)
  print(args.name)
  print(args.age)
  main()