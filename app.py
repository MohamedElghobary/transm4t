import requests

def testApI(text,lang):

    # Define the API endpoint URL
    api_url = "http://a270-34-73-231-86.ngrok.io/translate"  # Replace with your ngrok public URL

    # Define the input data as a dictionary
    data = {
        "text": f"{text}",
        "target_language": f"{lang}"  # Optional: Specify the target language (e.g., Arabic)
    }

    # Send a POST request to the API
    response = requests.post(api_url, json=data)

    # Check the response status code
    if response.status_code == 200:
        # If the request was successful (status code 200), parse the JSON response
        response_json = response.json()
        translated_text = response_json.get('translated_text', 'Translation not available')
        print("Translated text:", translated_text)
        return translated_text
    else:
        # If there was an error, print the status code and response content
        print("Error - Status Code:", response.status_code)
        print("Response Content:", response.content.decode('utf-8'))



testApI("Two things are infinite: the الكون and human stupidity; and I'm not sure about the universe.",'eng')
