"""Example module demonstrating NumPy-style docstrings."""


class Example:
    """Example class used to demonstrate API documentation generation."""

    def add(self, a: int, b: int) -> int:
        """Add two integers.

        Parameters
        ----------
        a : int
            First integer to add.
        b : int
            Second integer to add.

        Returns
        -------
        int
            The sum of the two integers.

        Notes
        -----
        Docstrings are rendered by mkdocstrings on the API page.
        """
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract two integers.

        Parameters
        ----------
        a : int
            Integer to subtract from.
        b : int
            Integer to subtract.

        Returns
        -------
        int
            The difference of the two integers.
        """
        return a - b
