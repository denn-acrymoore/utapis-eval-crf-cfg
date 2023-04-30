import os
import json
import re


def move_existing_tag_to_new_preprocessed_file(
    new_preprocessed_file_path, old_tagged_file_path, output_path
):
    """
    This code is used to combine the tag values of the manually tagged files
    and the word of the preprocessed files (using new preprocessing function)
    """
    new_preprocessed_list = os.listdir(new_preprocessed_file_path)
    new_preprocessed_list.sort()
    # print(new_preprocessed_list)

    old_tagged_list = os.listdir(old_tagged_file_path)
    old_tagged_list.sort()
    # print(old_tagged_list)

    if len(new_preprocessed_list) != len(old_tagged_list):
        print(
            "Error: Number of new preprocessed files and old tagged files aren't the same"
        )
        return

    for new_preprocessed_file, old_tagged_file in zip(
        new_preprocessed_list, old_tagged_list
    ):
        preprocessed_path = os.path.join(
            new_preprocessed_file_path, new_preprocessed_file
        )
        tagged_path = os.path.join(old_tagged_file_path, old_tagged_file)

        with open(preprocessed_path, "r") as fp:
            new_preprocessed_json = json.load(fp)

        with open(tagged_path, "r") as fp:
            old_tagged_json = json.load(fp)

        output_json = []

        for kal_preprocessed, kal_tagged in zip(new_preprocessed_json, old_tagged_json):
            temp_output_json = []
            temp_output_json.append(kal_tagged[0])
            temp_output_json.append(
                list(
                    zip(
                        [item[0] for item in kal_preprocessed[1]],
                        [item[1] for item in kal_tagged[1]],
                    )
                )
            )

            output_json.append(temp_output_json)

        output_file_name = re.search(
            r"^\(.+\)\s(.+)$", os.path.splitext(new_preprocessed_file)[0]
        )
        output_file_name = "(tagged) " + output_file_name.group(1) + ".json"
        output_file_name = os.path.join(output_path, output_file_name)

        with open(output_file_name, "w") as fp:
            json.dump(output_json, fp)

        print("File:", output_file_name, "created!")


if __name__ == "__main__":
    old_tagged_folder_name = (
        "(deprecated 30 Apr 2023 10.08) processed and manually tagged"
    )

    main_path = os.path.dirname(__file__)

    new_preprocessed_file_path = os.path.join(main_path, "processed")
    old_tagged_file_path = os.path.join(main_path, old_tagged_folder_name)
    output_path = os.path.join(main_path, "processed and manually tagged")

    move_existing_tag_to_new_preprocessed_file(
        new_preprocessed_file_path, old_tagged_file_path, output_path
    )
