import datetime

class Account:
    """
    Represents a bank account.

    Attributes:
        account_number (str): The account number.
        balance (float): The account balance.
        creation_date (datetime.date): The date the account was created.

    Methods:
        deposit(amount: float) -> None: Deposits funds into the account.
        withdraw(amount: float) -> None: Withdraws funds from the account.
        get_balance() -> float: Returns the account balance.
        get_account_info() -> str: Returns a string with account information.

    Raises:
        ValueError: If the amount is negative or if withdrawal exceeds balance.
    """

    def __init__(self, account_number: str, initial_balance: float = 0.0) -> None:
        """Initializes an Account instance.

        Args:
            account_number (str): The account number.
            initial_balance (float, optional): The initial balance. Defaults to 0.0.
        """
        self.account_number = account_number
        self.balance = initial_balance
        self.creation_date = datetime.date.today()

    def deposit(self, amount: float) -> None:
        """Deposits funds into the account.

        Args:
            amount (float): The amount to deposit.

        Raises:
            ValueError: If the amount is negative.
        """
        if amount < 0:
            raise ValueError("Deposit amount cannot be negative.")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        """Withdraws funds from the account.

        Args:
            amount (float): The amount to withdraw.

        Raises:
            ValueError: If the amount is negative or if withdrawal exceeds balance.
        """
        if amount < 0:
            raise ValueError("Withdrawal amount cannot be negative.")
        if amount > self.balance:
            raise ValueError("Insufficient funds.")
        self.balance -= amount

    def get_balance(self) -> float:
        """Returns the account balance.

        Returns:
            float: The account balance.
        """
        return self.balance

    def get_account_info(self) -> str:
        """Returns a string with account information.

        Returns:
            str: A string containing account information.
        """
        return (f"Account Number: {self.account_number}\n"
                f"Balance: ${self.balance:.2f}\n"
                f"Creation Date: {self.creation_date.strftime('%Y-%m-%d')}")


def main():
    """Main function to run the simple banking application CLI."""
    account_number = input("Enter account number: ")
    try:
        account = Account(account_number)
        while True:
            print("\nSelect operation:")
            print("1. Deposit")
            print("2. Withdraw")
            print("3. Check Balance")
            print("4. Account Info")
            print("5. Exit")

            choice = input("Enter choice(1/2/3/4/5): ")

            if choice == '1':
                amount = float(input("Enter amount to deposit: "))
                account.deposit(amount)
                print("Deposit successful.")
            elif choice == '2':
                amount = float(input("Enter amount to withdraw: "))
                account.withdraw(amount)
                print("Withdrawal successful.")
            elif choice == '3':
                print(f"Account balance: ${account.get_balance():.2f}")
            elif choice == '4':
                print(account.get_account_info())
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()