from automated_search import perform_search, list_templates, list_email_settings

# Get first available template
templates = list_templates()
if not templates:
    print("No templates found. Please create a template first.")
    exit(1)

template_id = templates[0].split(":")[0]
print(f"Using template ID: {template_id}")

# Get first available email setting
email_settings = list_email_settings()
if not email_settings:
    print("No email settings found. Please configure email settings first.")
    exit(1)

from_email = email_settings[0]['email']
print(f"Using email: {from_email}")

# Search terms for Spain
search_terms = [
    "desarrollador software españa",
    "programador senior españa",
    "desarrollador fullstack españa",
    "ingeniero software españa",
    "desarrollador backend españa"
]

# Run search
results, logs = perform_search(
    search_terms=search_terms,
    template_id=template_id,
    from_email=from_email,
    num_results=10,  # 10 results per term
    language='ES',
    enable_email_sending=True
)

# Print logs
print("\nSearch Logs:")
print("=" * 50)
for log in logs:
    print(log)

# Print results summary
print("\nSearch Results Summary:")
print("=" * 50)
print(f"Total terms searched: {len(search_terms)}")
print(f"Total leads found: {len(results)}")

# Print detailed results
print("\nDetailed Results:")
print("=" * 50)
for result in results:
    print(f"Email: {result['Email']}")
    print(f"Company: {result['Company']}")
    print(f"Source: {result['Source']}")
    print(f"Search Term: {result['Term']}")
    print("-" * 30) 