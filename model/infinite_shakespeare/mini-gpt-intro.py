# Open the file for reading 
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

print("length of dataset in characters: ", len(text))

# Print the first 1000 characters
print(text[:1000])
