import os
import time
import pandas as pd
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

DRIVER_PATH = os.getenv("DRIVER_PATH")
CSV_PATH = os.getenv("CSV_PATH")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

def init_browser():
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get("https://phy-x.net/module/physics/shielding/")
    print("Website loaded successfully")
    return driver

def login(driver, data):
    driver.get("https://phy-x.net/module/physics/shielding/")
    register_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//button[text()='Register']"))
    )
    register_button.click()
    print("Register button clicked")

    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()='Login']"))
    )
    login_button.click()
    print("Login button clicked")

    time.sleep(2)
    print("Time to login")


    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "user_mail"))
    )

    username_input = driver.find_element(By.XPATH, "//input[@label='Username']")
    username_input.send_keys(USERNAME)
    print("Username entered")

    password_input = driver.find_element(By.XPATH, "//input[@label='Password']")
    password_input.send_keys(PASSWORD)
    print("Password entered")


    authenticate_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()='Authenticate']"))
    )
    authenticate_button.click()
    print("Authenticate button clicked")
    print("Loading data\n", data.head(2))
    time.sleep(5)
    print("Page loaded, ready to fill data")

def enter_data(driver, data):
    index = 0
    while index < len(data):
        row = data.iloc[index]
        composition = row['Composition_Formula']
        density = row['Theoretical_Density']

        composition_id = f"formula{index}" if index > 0 else "formula"
        density_id = f"density{index}" if index > 0 else "density"
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, composition_id))
        ).send_keys(composition)
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, density_id))
        ).send_keys(str(density))

        index += 1

def initalizing_simulation(driver):
    try:
        analyze_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.primary.calculate"))
        )
        analyze_button.click()
        print("Analyze button clicked")
    except Exception as e:
        print(f"An error occurred while clicking the Analyze button: {e}")
    try:
        ok_button= WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH,"//button[text()='OK']"))
        )
        ok_button.click()
        print("OK button is clicked")
    except Exception as e:
        print(f"an error occurred while clicking OK button {e}")

def process_checkboxes(driver):
        
    try:
        all_checkboxes = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.XPATH, "//input[@type='checkbox' and @class='processor']"))
        )
        print(f"Found checkboxes: {len(all_checkboxes)}")

        for checkbox in all_checkboxes:
            if checkbox.is_selected():
                driver.execute_script("arguments[0].click();", checkbox)
                print("Checkbox deselected")

    except Exception as e:
        print(f"An error occurred while deselecting checkboxes: {e}")

    print("All checkboxes processed.")

    try:
        all_checkboxes = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.XPATH, "//input[@type='checkbox' and @class='processor']"))
        )
        print(f"Found checkboxes: {len(all_checkboxes)}")
        checkbox_ids_to_select = ['FNRCS']
        for checkbox in all_checkboxes:
            checkbox_id = checkbox.get_attribute('id')
            if checkbox_id in checkbox_ids_to_select:
                if not checkbox.is_selected():
                    driver.execute_script("arguments[0].click();", checkbox)
                    print(f"Checkbox with ID {checkbox_id} selected using JavaScript")
                else:
                    print(f"Checkbox with ID {checkbox_id} was already selected")
            else:
                if checkbox.is_selected():
                    driver.execute_script("arguments[0].click();", checkbox)
                    print(f"Checkbox with ID {checkbox_id} deselected")

    except Exception as e:
        print(f"An error occurred while processing checkboxes: {e}")

    print("Specific checkboxes processed.")
def accepting_terms(driver):
    try:
        accepted_checkbox = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "accepted"))
        )
        if not accepted_checkbox.is_selected():
            driver.execute_script("arguments[0].click();", accepted_checkbox)
            print("Checkbox 'accepted' selected using JavaScript")
        else:
            print("Checkbox 'accepted' was already selected")
    except Exception as e:
        print("Checkbox accepted was not selected", e)
        
    try:
        calculate_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'ui buttons main double checkbox')]//button"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", calculate_button)
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable(calculate_button)
        ).click()
        print("Calculate button clicked")
    except Exception as e:
        print(f"An error occurred while clicking the Calculate button: {e}")
    print("Calculate button clicked")
def run_and_exprt_simulation(driver):
    try:
        performance_checkbox = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "performance-checkbox"))
        )
        if not performance_checkbox.is_selected():
            driver.execute_script("arguments[0].click();", performance_checkbox)
            print("Performance checkbox selected using JavaScript")
        else:
            print("Performance checkbox was already selected")
    except Exception as e:
        print(f"An error occurred while selecting the performance checkbox: {e}")

    try:
        ok_button_new_page = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'ui buttons main single foot')]//button"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", ok_button_new_page)
        start_time = time.time()
        print("start_time", start_time)
        ok_button_new_page.click()
        print("OK button clicked on the new page")
    except Exception as e:
        print(f"An error occurred while clicking the OK button on the new page: {e}")

    print("Performance checkbox and OK button on the new page processed.")

    try:
        export_button = WebDriverWait(driver, 600).until(
            EC.presence_of_element_located((By.XPATH, "//button[text()='Export to Excel']"))
        )
        end_time = time.time()
        print("end time", end_time)
        export_button.click()
        simulation_time = end_time - start_time
        print("simulation time", simulation_time)
        print(f"Simulation completed. Time taken: {simulation_time%60:.2f} seconds")
        print("data has been exported")
        
    except Exception as e:
        print(f"An error occurred while waiting for the 'Export to Excel' button: {e}")
def main():
    data = pd.read_csv(CSV_PATH)
    print(data)
    # data = data[1:10]  
    # print(data[3])
    print(f"Processing {len(data)} rows of data.")

    driver = init_browser()

    try:
        time.sleep(2)
        login(driver, data)
        enter_data(driver, data)
        time.sleep(2)
        initalizing_simulation(driver)
        process_checkboxes(driver)
        accepting_terms(driver)
        run_and_exprt_simulation(driver)

    finally:
        input('Press enter to close the browser')
        driver.quit()

if __name__ == "__main__":
    main()
