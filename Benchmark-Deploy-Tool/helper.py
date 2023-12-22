def convert_to_number(arg):
    if arg.endswith('ms') or arg.endswith('MB'):
        return int(arg[:-2]), arg[-2:]
    if arg.endswith('s') or arg.endswith('m'):
        return int(arg[:-1]), arg[-1:]
    return int(arg), ''



