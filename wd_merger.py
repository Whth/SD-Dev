import pathlib
from typing import List, Dict

from modules.file_manager import explore_folder


class WildCardMerger:
    @staticmethod
    def merge(file_lists: List[str], output_path: str):
        path = pathlib.Path(output_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for file in file_lists:
                file_to_merge = pathlib.Path(file)
                with file_to_merge.open("r", encoding="utf-8") as f1:
                    f.write(f1.read())
                    f.write("\n")
                file_to_merge.unlink(missing_ok=True)

    @staticmethod
    def extract_assembly_pairs(file_lists: List[str]) -> Dict[str, List[str]]:
        names: List[str] = list(map(lambda x: pathlib.Path(x).stem, file_lists))
        tokenized_names: List[List[str]] = [name.split("_") for name in names]
        paired_table: Dict[str, List[str]] = {}
        for tokenized_name, file_path in zip(tokenized_names, file_lists):
            if tokenized_name[0] in paired_table:
                paired_table.get(tokenized_name[0]).append(file_path)
            else:
                paired_table[tokenized_name[0]] = [file_path]
        return paired_table

    def assembly(self, dir_path: str, output_dir_path: str):
        out_dir = pathlib.Path(output_dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not out_dir.is_dir():
            raise ValueError(f"Output directory {out_dir} does not exist.")
        file_list: List[str] = explore_folder(root_path=dir_path)

        file_list = list(filter(lambda x: x.endswith(".txt"), file_list))
        paired_table = self.extract_assembly_pairs(file_list)
        for out_put_name, file_list in paired_table.items():
            self.merge(file_list, str(out_dir.joinpath(out_put_name + ".txt")))


if __name__ == "__main__":
    merger = WildCardMerger()
    merger.assembly(
        dir_path=r"L:\pycharm projects\chatBotComponents\extensions\SD-Dev\asset\wildcard",
        output_dir_path=r"L:\pycharm projects\chatBotComponents\extensions\SD-Dev\asset\wildcard\temp",
    )
