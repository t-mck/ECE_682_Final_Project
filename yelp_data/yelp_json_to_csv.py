# -*- coding: utf-8 -*-
"""Convert the Yelp Dataset Challenge dataset from json format to csv.
For more information on the Yelp Dataset Challenge please visit http://yelp.com/dataset_challenge
"""
import argparse
import collections
import csv
import simplejson as json


def remove_commas_and_newlines(line:str):
    text_loc = line.find("\"text\":")
    date_loc = line.find(",\"date\":")
    text = line[text_loc:date_loc]

    pre_text = line[0:text_loc]
    post_text = line[date_loc:len(line)]

    new_text = text
    comma_found = True
    while comma_found:
        new_text = new_text.replace(",", "")
        if new_text.find(",") != -1:
            comma_found = True
        else:
            comma_found = False

    new_text2 = new_text
    # newline_found = True
    # while newline_found:
    #     new_text2 = new_text2.replace("\\n", "")
    #     if new_text2.find("\\n") != -1:
    #         newline_found = True
    #     else:
    #         newline_found = False

    new_text3 = new_text2
    semicolon_found = True
    while semicolon_found:
        new_text3 = new_text3.replace(";", "")
        if new_text3.find(";") != -1:
            semicolon_found = True
        else:
            semicolon_found = False

    new_line = pre_text + new_text3 + post_text
    return new_line

def write_lines_to_file(lc_list, column_names, csv_file_path):
    with open(csv_file_path, 'w+') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        for line in lc_list:
            csv_file.writerow(get_row(line, column_names))

def read_and_write__multiple_csv_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    num_files_index = 0
    max_lines = 200000
    lc_list = []
    with open(json_file_path) as fin:
        for line in fin:
            new_line = remove_commas_and_newlines(line)
            line_contents = json.loads(new_line)
            lc_list.append(line_contents)
            if len(lc_list) >= max_lines:
                csv_file_path_to_use = f'{csv_file_path}_{num_files_index}_.csv'
                write_lines_to_file(lc_list, column_names, csv_file_path_to_use)
                lc_list.clear()
                num_files_index += 1
            x = 1
        if len(lc_list) > 0:
            csv_file_path_to_use = f'{csv_file_path}_{num_files_index}_.csv'
            write_lines_to_file(lc_list, column_names, csv_file_path_to_use)
            lc_list.clear()
            num_files_index += 1
    x = 1


def read_and_write_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    read_and_write__multiple_csv_file(json_file_path, csv_file_path, column_names)

    # with open(csv_file_path, 'w+') as fout:
    #     csv_file = csv.writer(fout)
    #     csv_file.writerow(list(column_names))
    #     with open(json_file_path) as fin:
    #         for line in fin:
    #             new_line = remove_commas_and_newlines(line)
    #             line_contents = json.loads(new_line)
    #             csv_file.writerow(get_row(line_contents, column_names))


def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    column_names = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                set(get_column_names(line_contents).keys())
            )
    return column_names


def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.
    Example:
        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        will return: ['a.b', 'a.c']
    These will be the column names for the eventual csv file.
    """
    column_names = []
    for k, v in line_contents.items(): # iteritems():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                get_column_names(v, column_name).items()
            )
        else:
            column_names.append((column_name, v))
    return dict(column_names)


def get_nested_value(d, key):
    """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.

    Example:
        d = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        key = 'a.b'
        will return: 2

    """
    if '.' not in key:
        if key not in d:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)


def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
            line_contents,
            column_name,
        )
        if isinstance(line_value, str):
            row.append('{0}'.format(line_value.encode('utf-8')))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row


if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""

    parser = argparse.ArgumentParser(
        description='Convert Yelp Dataset Challenge data from JSON format to CSV.',
    )

    parser.add_argument(
        'json_file',
        type=str,
        help='The json file to convert.',
    )

    args = parser.parse_args()

    json_file = args.json_file
    csv_file = '{0}.csv'.format(json_file.split('.json')[0])

    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)
