import os
from dotenv import load_dotenv
from automated_search import (
    list_templates,
    list_email_settings,
    create_template,
    perform_search
)

def test_database_connection():
    """Test if we can connect to the database"""
    try:
        templates = list_templates()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

def test_list_templates():
    """Test listing templates"""
    try:
        templates = list_templates()
        print(f"✅ Found {len(templates)} templates:")
        for template in templates:
            print(f"  - {template}")
        return True
    except Exception as e:
        print(f"❌ List templates failed: {str(e)}")
        return False

def test_list_email_settings():
    """Test listing email settings"""
    try:
        settings = list_email_settings()
        print(f"✅ Found {len(settings)} email settings:")
        for setting in settings:
            print(f"  - {setting['name']} ({setting['email']})")
        return True
    except Exception as e:
        print(f"❌ List email settings failed: {str(e)}")
        return False

def test_create_template():
    """Test template creation"""
    try:
        template = create_template(
            template_name="Test Template",
            subject="Test Subject",
            body_content="Test Content",
            language="ES"
        )
        print(f"✅ Created template: {template}")
        return True
    except Exception as e:
        print(f"❌ Create template failed: {str(e)}")
        return False

def test_search():
    """Test search functionality"""
    try:
        # Get first template
        templates = list_templates()
        if not templates:
            print("❌ No templates available for testing search")
            return False
            
        template_id = templates[0].split(":")[0]
        
        # Get first email setting
        settings = list_email_settings()
        if not settings:
            print("❌ No email settings available for testing search")
            return False
            
        from_email = settings[0]['email']
        
        # Perform test search
        results, logs = perform_search(
            search_terms=["test developer"],
            template_id=template_id,
            from_email=from_email,
            num_results=2,
            enable_email_sending=False  # Disable email sending for test
        )
        
        print("✅ Search completed successfully")
        print("\nLogs:")
        for log in logs:
            print(f"  {log}")
            
        print("\nResults:")
        for result in results:
            print(f"  Found: {result.get('Email')} from {result.get('Company')}")
            
        return True
    except Exception as e:
        print(f"❌ Search test failed: {str(e)}")
        return False

if __name__ == "__main__":
    load_dotenv()
    
    print("\n🔍 Testing Automated Search Functions\n")
    
    # Run tests
    tests = [
        ("Database Connection", test_database_connection),
        ("List Templates", test_list_templates),
        ("List Email Settings", test_list_email_settings),
        ("Create Template", test_create_template),
        ("Search Functionality", test_search)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        success = test_func()
        results.append((test_name, success))
        print("-" * 50)
    
    # Print summary
    print("\n📊 Test Summary:")
    for test_name, success in results:
        status = "✅ Passed" if success else "❌ Failed"
        print(f"{status} - {test_name}") 