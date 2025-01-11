from streamlit_app import manual_search, db_session

def test_manual_search():
    print("Starting manual search test...")
    with db_session() as session:
        print("Database session created")
        results = manual_search(
            session=session,
            search_terms=['software engineer'],
            num_results=10,
            ignore_previously_fetched=True,
            optimize_english=False,
            optimize_spanish=False,
            shuffle_keywords=False,
            language='ES',
            enable_email_sending=True,
            log_container=None,
            from_email=None,
            reply_to=None,
            email_template=None
        )
        print("Search Results:", results)

if __name__ == "__main__":
    test_manual_search() 