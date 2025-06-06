/*
 * Banking Application Console
 *
 * This program simulates a simple banking application console.
 * It allows users to perform basic banking operations such as creating accounts,
 * depositing, withdrawing, and checking balances.
 *
 * Input: User commands via console input.
 * Output: Account information and transaction results displayed on the console.
 *
 * Usage: Run the program from the command line. Follow the on-screen prompts.
 */

import java.util.HashMap;
import java.util.InputMismatchException;
import java.util.Scanner;

public class BankingApplication {

    private static HashMap<Integer, Account> accounts = new HashMap<>();
    private static int nextAccountId = 1;

    /**
     * Main method to run the banking application.
     * @param args Command line arguments (not used).
     */
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        while (true) {
            displayMenu();
            try {
                int choice = scanner.nextInt();
                scanner.nextLine(); // Consume newline

                switch (choice) {
                    case 1:
                        createAccount(scanner);
                        break;
                    case 2:
                        deposit(scanner);
                        break;
                    case 3:
                        withdraw(scanner);
                        break;
                    case 4:
                        checkBalance(scanner);
                        break;
                    case 5:
                        System.out.println("Exiting...");
                        return;
                    default:
                        System.out.println("Invalid choice. Please try again.");
                }
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a number.");
                scanner.nextLine(); // Consume invalid input
            }
        }
    }

    /**
     * Displays the main menu of the banking application.
     */
    private static void displayMenu() {
        System.out.println("\nBanking Application Menu:");
        System.out.println("1. Create Account");
        System.out.println("2. Deposit");
        System.out.println("3. Withdraw");
        System.out.println("4. Check Balance");
        System.out.println("5. Exit");
        System.out.print("Enter your choice: ");
    }

    /**
     * Creates a new account.
     * @param scanner Scanner to read user input.
     */
    private static void createAccount(Scanner scanner) {
        System.out.print("Enter initial deposit amount: ");
        double balance;
        try {
            balance = scanner.nextDouble();
        } catch (InputMismatchException e) {
            System.out.println("Invalid input. Please enter a number.");
            scanner.nextLine();
            return;
        }

        if (balance <= 0) {
            System.out.println("Initial deposit must be greater than zero.");
            return;
        }
        scanner.nextLine(); // Consume newline

        Account account = new Account(nextAccountId++, balance);
        accounts.put(account.getId(), account);
        System.out.println("Account created successfully. Account ID: " + account.getId());
    }

    /**
     * Deposits money into an existing account.
     * @param scanner Scanner to read user input.
     */
    private static void deposit(Scanner scanner) {
        int accountId = getAccountId(scanner);
        if (accountId == -1) return;

        System.out.print("Enter deposit amount: ");
        double amount;
        try {
            amount = scanner.nextDouble();
        } catch (InputMismatchException e) {
            System.out.println("Invalid input. Please enter a number.");
            scanner.nextLine();
            return;
        }

        if (amount <= 0) {
            System.out.println("Deposit amount must be greater than zero.");
            return;
        }

        Account account = accounts.get(accountId);
        account.deposit(amount);
        System.out.println("Deposit successful.");
    }

    /**
     * Withdraws money from an existing account.
     * @param scanner Scanner to read user input.
     */
    private static void withdraw(Scanner scanner) {
        int accountId = getAccountId(scanner);
        if (accountId == -1) return;

        System.out.print("Enter withdrawal amount: ");
        double amount;
        try {
            amount = scanner.nextDouble();
        } catch (InputMismatchException e) {
            System.out.println("Invalid input. Please enter a number.");
            scanner.nextLine();
            return;
        }
        if (amount <= 0) {
            System.out.println("Withdrawal amount must be greater than zero.");
            return;
        }

        Account account = accounts.get(accountId);
        if (!account.withdraw(amount)) {
            System.out.println("Insufficient balance.");
        } else {
            System.out.println("Withdrawal successful.");
        }
    }

    /**
     * Checks the balance of an existing account.
     * @param scanner Scanner to read user input.
     */
    private static void checkBalance(Scanner scanner) {
        int accountId = getAccountId(scanner);
        if (accountId == -1) return;

        Account account = accounts.get(accountId);
        System.out.println("Account balance: " + account.getBalance());
    }

    /**
     * Gets the account ID from the user and validates it.
     * @param scanner Scanner to read user input.
     * @return The account ID, or -1 if the ID is invalid.
     */
    private static int getAccountId(Scanner scanner) {
        System.out.print("Enter account ID: ");
        int accountId;
        try {
            accountId = scanner.nextInt();
        } catch (InputMismatchException e) {
            System.out.println("Invalid input. Please enter a number.");
            scanner.nextLine();
            return -1;
        }
        if (!accounts.containsKey(accountId)) {
            System.out.println("Invalid account ID.");
            return -1;
        }
        return accountId;
    }

    /**
     * Inner class representing a bank account.
     */
    static class Account {
        private int id;
        private double balance;

        /**
         * Constructs a new Account object.
         * @param id The account ID.
         * @param balance The initial balance.
         */
        public Account(int id, double balance) {
            this.id = id;
            this.balance = balance;
        }

        public int getId() {
            return id;
        }

        public double getBalance() {
            return balance;
        }

        /**
         * Deposits money into the account.
         * @param amount The amount to deposit.
         */
        public void deposit(double amount) {
            balance += amount;
        }

        /**
         * Withdraws money from the account.
         * @param amount The amount to withdraw.
         * @return True if the withdrawal was successful, false otherwise.
         */
        public boolean withdraw(double amount) {
            if (balance >= amount) {
                balance -= amount;
                return true;
            }
            return false;
        }
    }
}