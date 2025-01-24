import sys
sys.path.append('.')
from streamlit_app import manual_search, db_session

class MockContainer:
    def markdown(self, text, unsafe_allow_html=False):
        print(text)
    def text(self, text):
        print(text)

def test_basic_search():
    print('Test 1: Basic Search')
    with db_session() as session:
        mock_container = MockContainer()
        result = manual_search(
            session=session,
            terms=['software developer'],
            num_results=2,
            language='ES',
            ignore_previously_fetched=True,
            optimize_english=False,
            optimize_spanish=False,
            shuffle_keywords_option=False,
            enable_email_sending=False,
            log_container=mock_container
        )
        print('Basic Search Results:', result)

def test_optimized_search():
    print('Test 2: Optimized Search (English)')
    with db_session() as session:
        mock_container = MockContainer()
        result = manual_search(
            session=session,
            terms=['software engineer'],
            num_results=2,
            language='EN',
            ignore_previously_fetched=True,
            optimize_english=True,
            optimize_spanish=False,
            shuffle_keywords_option=True,
            enable_email_sending=False,
            log_container=mock_container
        )
        print('Optimized Search Results:', result)

if __name__ == '__main__':
    print('Starting tests...')
    test_basic_search()
    test_optimized_search()
    print('Tests completed.')