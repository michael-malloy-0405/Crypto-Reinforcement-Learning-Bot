import time
import csv
import asyncio
import os  # Added to check for the stop.txt file
from playwright.async_api import async_playwright

# Enhanced market_cap_toString function
def price_toFloat(price):
    if ',' in price:
        price = price.replace(',', '')
        return float(price)

def market_cap_toFloat(market_cap_string):
    if 'T' in market_cap_string:
        return float(market_cap_string[1:5]) * (10**12)
    
def trading_volume_toFloat(trading_volume_string):
    if 'B' in trading_volume_string:
        return float(trading_volume_string[1:6]) * (10**9)
    
def supply_toFloat(supply):
    if 'M' in supply:
        return float(supply[1:6]) * (10**6)

# Function to write the CSV header if file is empty
def write_csv_header(file_name):
    try:
        with open(file_name, mode="a", newline="") as file:
            if file.tell() == 0:  # If the file is empty
                writer = csv.writer(file)
                # Write the header
                writer.writerow(["price", "market_cap", "Fully Diluted Market Cap", "volume", "Volume/Market Cap", "Circulating Supply", "Max Supply"])
    except Exception as e:
        print(f"Error writing CSV header: {e}")

# Function to check if the stop file exists
def should_stop():
    return os.path.exists("stop.txt")

# Main function
async def main():
    # Define the CSV file name
    csv_file_name = "price_data.csv"
    write_csv_header(csv_file_name)  # Ensure the CSV file has a header

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Go to the website
        url = "https://www.tradingview.com/symbols/BTCUSD/?exchange=CRYPTO"
        await page.goto(url)

        # Wait for the elements to appear
        await page.wait_for_selector("span.last-JWoJqCpY.js-symbol-last")
        await page.wait_for_selector("div.apply-overflow-tooltip.value-GgmpMpKr")  # Wait for market cap div

        while True:
            if should_stop():  # Check for the stop file
                print("Stopping the script...")
                break  # Break out of the loop to stop the script

            # Get the current value of the price
            price = await page.inner_text("span.last-JWoJqCpY.js-symbol-last")
            price = price_toFloat(price)
            market_cap_elements = await page.query_selector_all("div.apply-overflow-tooltip.value-GgmpMpKr")

            if len(market_cap_elements) >= 2:
                # Get the regular market cap (first element)
                market_cap = await market_cap_elements[0].inner_text()
                # Get the fully diluted market cap (second element)
                fully_diluted_cap = await market_cap_elements[1].inner_text()

                trading_volume = await market_cap_elements[2].inner_text()
                volume_over_market_cap = await market_cap_elements[3].inner_text()
                circulating_supply = await market_cap_elements[5].inner_text()
                max_supply = await market_cap_elements[6].inner_text()

                # Convert to numeric values
                market_cap_float = market_cap_toFloat(market_cap)
                fully_diluted_cap_float = market_cap_toFloat(fully_diluted_cap)
                trading_volume_float = trading_volume_toFloat(trading_volume)
                circulating_supply_float = supply_toFloat(circulating_supply)
                max_supply_float = supply_toFloat(max_supply)

            else:
                print("Not enough market cap elements found.")
                market_cap_float = None
                fully_diluted_cap_float = None
                trading_volume = None

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp

            # Write the timestamp, price, and market caps to the CSV file
            try:
                with open(csv_file_name, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, price, market_cap_float, fully_diluted_cap_float, trading_volume_float, volume_over_market_cap, circulating_supply_float, max_supply_float])
                print(f"Data written: {timestamp}, {price}, {market_cap_float}, {fully_diluted_cap_float}, {trading_volume_float}, {volume_over_market_cap}, {circulating_supply_float}, {max_supply_float}")
            except Exception as e:
                print(f"Error writing to CSV: {e}")

            # Wait a bit before checking again
            time.sleep(0.8)

        # Close the browser
        await browser.close()

# Run the script
if __name__ == "__main__":
    asyncio.run(main())
