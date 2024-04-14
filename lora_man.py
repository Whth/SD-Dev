import warnings
from typing import List, Optional, Self, Union, Dict, Any, Literal

from pydantic import BaseModel


class LoraUnit(BaseModel):
    index: int
    name: str
    weight: float = 1.0

    def __str__(self):
        return f"<lora:{self.name}:{self.weight}>"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, LoraUnit):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False


class LoraManager(BaseModel):
    container: List[LoraUnit] = []
    lora_pool: List[str] = []

    def clean(self) -> Self:
        """
        A function to remove all LoraUnits from the container.
        Returns:

        """
        self.container.clear()
        return self

    def reasign_pool(self, new_pool: List[str]):
        """
        A function to reassign the pool with a new list of strings.

        Parameters:
            new_pool (List[str]): The new list of strings to assign to the pool.

        Returns:
            None
        """
        self.lora_pool = new_pool
        self.update_index()

    def update_index(self) -> Self:
        """
        A function to update the index based on the name in the container using the lora pool.
        """
        for unit in self.container:
            if unit.name not in self.lora_pool:
                warnings.warn(f"Lora {unit.name} is not in the pool, Skipping it")
                continue
            unit.index = self.lora_pool.index(unit.name)
        return self

    def use(self, identifier: Union[int, str], weight: Optional[float] = 1.0) -> Self:
        """
        A method to add a LoraUnit to the container based on the provided identifier and weight.

        Parameters:
            identifier (Union[int, str]): An integer index or string name to identify the LoraUnit.
            weight (Optional[float]): The weight of the LoraUnit. Defaults to 1.0.

        Returns:
            Self: Returns the instance of the class for method chaining.
        """
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.lora_pool):
                self.container.append(LoraUnit(index=identifier, name=self.lora_pool[identifier], weight=weight))
            else:
                warnings.warn(f"Index {identifier} is out of range,available range is [0, {len(self.lora_pool) - 1}]")
        elif isinstance(identifier, str):
            if identifier in self.lora_pool:
                self.container.append(LoraUnit(index=self.lora_pool.index(identifier), name=identifier, weight=weight))
            else:
                warnings.warn(f"Lora {identifier} is not in the pool")
        return self

    def remove(self, identifier: Union[int, str]) -> Self:
        """
        A method to remove a LoraUnit from the container based on the provided identifier.
        """
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.lora_pool):
                self.container = [unit for unit in self.container if unit.index != identifier]
            else:
                warnings.warn(f"Index {identifier} is out of range,available range is [0, {len(self.lora_pool) - 1}]")
        elif isinstance(identifier, str):
            if identifier in self.lora_pool:
                self.container = [unit for unit in self.container if unit.name != identifier]
            else:
                warnings.warn(f"Lora {identifier} is not in the pool")
        return self

    def dedup(self) -> Self:
        """
        A method to remove duplicate LoraUnits from the container.
        """
        self.container.reverse()
        self.container = list(set(self.container))
        return self

    def format(self) -> str:
        """
        A method that formats the container by joining all elements with a comma.
        """
        return ",".join([str(unit) for unit in self.container])

    def dump_container(self) -> List[Dict[str, Any]]:
        return [unit.dict() for unit in self.container]

    def parse_container(
        self, container_obj: List[Dict[str, Any]], strategy: Literal["merge", "replace"] = "replace"
    ) -> Self:
        if strategy == "merge":
            for unit in container_obj:
                self.container.append(LoraUnit(**unit))
        elif strategy == "replace":
            self.container = [LoraUnit(**unit) for unit in container_obj]
        else:
            raise ValueError("straight must be 'merge' or 'replace'")
        return self


if __name__ == "__main__":
    lora_pool = ["lora1", "lora2", "lora3"]
    lora_manager = LoraManager(lora_pool=lora_pool)
    print(lora_manager.use("lora1", weight=0.5).use("lora2", weight=0.5).use("lora3", weight=0.5).format())
    lora_manager.parse_container(lora_manager.dump_container(), strategy="merge")
    print(lora_manager.container)
    lora_manager.container[1].index = 5

    print(lora_manager.container)

    lora_manager.dedup()

    print(lora_manager.container)
    lora_manager.reasign_pool(lora_pool)
    print(lora_manager.lora_pool)

    lora_pool.append("lora4")

    print(lora_manager.lora_pool)
