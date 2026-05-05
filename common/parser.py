from os import listdir
import yaml


#################################################################
# general yml config parse


def get_params(config_file):
    with open(config_file, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


#################################################################
# model config parse


def create_class_from_dict(class_name, attributes):
    # Define a new class with the given name and attributes
    return type(class_name, (object,), attributes)


def parse_dict(data):
    attributes = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively create a class for nested dictionaries
            attributes[key] = parse_dict(value)
        else:
            attributes[key] = value
    return attributes


def yaml_cfg_to_class(yaml_file, name, content_name):
    # Read the YAML file
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    # Define the class name
    class_name = yaml_content.get(name)

    # Extract and parse attributes from YAML content
    attributes = parse_dict(yaml_content.get(content_name, {}))
    attributes["class_name"] = class_name
    # Dynamically create the class
    new_class = create_class_from_dict(class_name, attributes)

    return new_class


#################################################################
# file parsing functions


def parse_dir_for_X_file(directory: str, x_ending: str):
    wanted_files = []
    files = listdir(directory)
    for f in files:
        if f.endswith(x_ending):
            wanted_files.append(f)
    print("Found {} {} files in dir: {}".format(len(wanted_files), x_ending, directory))
    return wanted_files
