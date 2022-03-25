from dataclasses import fields

class TaggedRoster:
    """
    Assuming that subclasses are pydantic's dataclass. All the fields must be Optional
    w/ None as default value. This doesn't support changing selected field/value.
    """

    @property
    def value(self):
        selected_fields = [
            field.name for field in fields(self) if getattr(self, field.name, None)
        ]
        assert (
            len(selected_fields) == 1
        ), f"{self} Expecting one selected field, got {selected_fields}"
        return getattr(self, selected_fields[0])