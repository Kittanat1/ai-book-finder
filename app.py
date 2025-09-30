import streamlit as st
import requests
import time
import math
from collections import Counter

st.set_page_config(page_title="AI Book Finder", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stButton button { width: 100%; }
    .metric-card { background: #f8fafc; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# === IR Functions ===
def tokenize(text):
    if not text:
        return []
    return text.lower().replace(',', ' ').replace('.', ' ').split()

def build_tfidf(documents):
    vocab = set()
    for doc in documents:
        vocab.update(tokenize(doc))
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
    
    N = len(documents)
    idf = {}
    for word in vocab:
        df = sum(1 for doc in documents if word in tokenize(doc))
        idf[word] = math.log((N + 1) / (df + 1)) + 1
    
    vectors = []
    for doc in documents:
        tokens = tokenize(doc)
        tf = Counter(tokens)
        vector = [0] * len(vocab)
        for word, count in tf.items():
            if word in vocab_index:
                idx = vocab_index[word]
                vector[idx] = count * idf.get(word, 0)
        vectors.append(vector)
    
    return vectors, vocab_index, idf

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def search_with_ir(query, books, doc_vectors, vocab_index, idf):
    query_tokens = tokenize(query)
    query_tf = Counter(query_tokens)
    query_vector = [0] * len(vocab_index)
    
    for word, count in query_tf.items():
        if word in vocab_index:
            idx = vocab_index[word]
            query_vector[idx] = count * idf.get(word, 0)
    
    scores = []
    for i, doc_vec in enumerate(doc_vectors):
        cos_score = cosine_similarity(query_vector, doc_vec)
        
        # Field Boosting
        title_tokens = tokenize(books[i]['title'])
        author_tokens = tokenize(books[i]['author'])
        
        title_match = sum(3 for token in query_tokens if token in title_tokens)
        author_match = sum(2 for token in query_tokens if token in author_tokens)
        
        final_score = cos_score * 4.0 + title_match + author_match
        scores.append((books[i], final_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [book for book, score in scores if score > 0]

def recommend_similar(book_idx, books, doc_vectors, top_n=4):
    if book_idx < 0 or book_idx >= len(doc_vectors):
        return []
    
    base_vec = doc_vectors[book_idx]
    scores = []
    
    for i, doc_vec in enumerate(doc_vectors):
        if i != book_idx:
            sim = cosine_similarity(base_vec, doc_vec)
            scores.append((books[i], sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [book for book, _ in scores[:top_n]]

# === Session State ===
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'books' not in st.session_state:
    st.session_state.books = []
if 'loaded_initial' not in st.session_state:
    st.session_state.loaded_initial = False

def fetch_books_from_google(search_queries):
    all_books = []
    book_id = 1
    
    for query in search_queries:
        try:
            url = f'https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=10&langRestrict=en'
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                for item in items:
                    info = item.get('volumeInfo', {})
                    image_links = info.get('imageLinks', {})
                    
                    if image_links.get('thumbnail'):
                        all_books.append({
                            'id': f'b{book_id}',
                            'title': info.get('title', 'Unknown'),
                            'author': info.get('authors', ['Unknown'])[0] if info.get('authors') else 'Unknown',
                            'category': info.get('categories', ['General'])[0] if info.get('categories') else 'General',
                            'year': int(info.get('publishedDate', '2020')[:4]) if info.get('publishedDate') else 2020,
                            'rating': round(info.get('averageRating', 4.0), 1),
                            'description': (info.get('description', 'No description')[:300] + '...') if info.get('description') else 'No description',
                            'cover': image_links['thumbnail'].replace('http:', 'https:'),
                            'preview': info.get('previewLink', '')
                        })
                        book_id += 1
                        
                    if len(all_books) >= 100:
                        return all_books
            
            time.sleep(0.3)
        except:
            continue
    
    return all_books

if not st.session_state.loaded_initial:
    with st.spinner('กำลังโหลดหนังสือและสร้าง '):
        queries = [
            'Python programming', 'JavaScript', 'React', 'Data Science', 'Machine Learning',
            'Business strategy', 'Leadership', 'Finance', 'Marketing', 'Self improvement',
            'Fiction bestseller', 'Mystery', 'Romance', 'Science fiction', 'Fantasy',
            'History', 'Biography', 'Psychology', 'Philosophy', 'Art'
        ]
        st.session_state.books = fetch_books_from_google(queries)
        
        documents = [
            f"{b['title']} {b['author']} {b['category']} {b['description']}" 
            for b in st.session_state.books
        ]
        
        vectors, vocab_idx, idf_scores = build_tfidf(documents)
        
        st.session_state.doc_vectors = vectors
        st.session_state.vocab_index = vocab_idx
        st.session_state.idf = idf_scores
        st.session_state.loaded_initial = True

# === UI ===
st.title("📚 Book Finder")
st.markdown("**Powered by TF-IDF, Vector Space Model & Cosine Similarity**")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("หนังสือทั้งหมด", len(st.session_state.books))
with col_info2:
    st.metric("Vocabulary Size", len(st.session_state.vocab_index))
with col_info3:
    st.metric("รายการโปรด", len(st.session_state.favorites))

st.markdown("---")

popular_searches = [
    "Python", "JavaScript", "React", "Machine Learning", "Data Science",
    "Business", "Leadership", "Finance", "Fiction", "Mystery"
]

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    query = st.text_input("🔍 ค้นหา", placeholder="พิมพ์คำค้นหา...")

with col2:
    category_list = ["ทั้งหมด"] + sorted(list(set([b['category'] for b in st.session_state.books])))
    category = st.selectbox("หมวดหมู่", category_list)

with col3:
    sort_by = st.selectbox("เรียง", ["Score", "คะแนน", "ปีล่าสุด"])

if query:
    st.markdown("**💡 คำแนะนำ:**")
    suggestions = [s for s in popular_searches if query.lower() in s.lower()]
    if suggestions:
        cols = st.columns(len(suggestions[:5]))
        for i, suggestion in enumerate(suggestions[:5]):
            with cols[i]:
                if st.button(suggestion, key=f"sug_{i}"):
                    query = suggestion
                    st.rerun()

col_search, col_fav, col_all = st.columns([1, 1, 1])

with col_search:
    search_clicked = st.button("🔎 ค้นหา", use_container_width=True)

with col_fav:
    if st.button(f"❤️ รายการโปรด ({len(st.session_state.favorites)})", use_container_width=True):
        st.session_state.display_books = [b for b in st.session_state.books if b['id'] in st.session_state.favorites]
        st.rerun()

with col_all:
    if st.button("📚 แสดงทั้งหมด", use_container_width=True):
        st.session_state.display_books = st.session_state.books
        st.rerun()

if search_clicked and query:
    with st.spinner('กำลังค้นหา'):
        filtered = search_with_ir(
            query, 
            st.session_state.books,
            st.session_state.doc_vectors,
            st.session_state.vocab_index,
            st.session_state.idf
        )
        
        if category != "ทั้งหมด":
            filtered = [b for b in filtered if b['category'] == category]
        
        if sort_by == "คะแนน":
            filtered.sort(key=lambda x: x['rating'], reverse=True)
        elif sort_by == "ปีล่าสุด":
            filtered.sort(key=lambda x: x['year'], reverse=True)
        
        st.session_state.display_books = filtered
        st.success(f"พบ {len(filtered)} เล่มด้วย IR Ranking")

if 'display_books' not in st.session_state:
    st.session_state.display_books = st.session_state.books

st.markdown("---")
st.markdown(f"### 📖 แสดง {len(st.session_state.display_books)} เล่ม")

if st.session_state.display_books:
    cols_per_row = 5
    books = st.session_state.display_books
    
    for i in range(0, len(books), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(books):
                book = books[i + j]
                with col:
                    st.image(book['cover'], use_column_width=True)
                    st.markdown(f"**{book['title'][:30]}{'...' if len(book['title']) > 30 else ''}**")
                    st.caption(f"👤 {book['author'][:20]}")
                    st.caption(f"📚 {book['category']}")
                    st.caption(f"📅 {book['year']} | ⭐ {book['rating']}")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("📖", key=f"view_{book['id']}", help="ดู"):
                            st.session_state.selected_book = book
                            book_idx = next((i for i, b in enumerate(st.session_state.books) if b['id'] == book['id']), -1)
                            st.session_state.selected_idx = book_idx
                            st.rerun()
                    with col_btn2:
                        is_fav = book['id'] in st.session_state.favorites
                        if st.button("❤️" if is_fav else "🤍", key=f"fav_{book['id']}"):
                            if is_fav:
                                st.session_state.favorites.remove(book['id'])
                            else:
                                st.session_state.favorites.append(book['id'])
                            st.rerun()
else:
    st.info("ไม่พบหนังสือ ลองคำค้นอื่น")

if 'selected_book' in st.session_state:
    book = st.session_state.selected_book
    with st.expander("📖 รายละเอียดหนังสือ", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(book['cover'], width=200)
        with col2:
            st.markdown(f"## {book['title']}")
            st.markdown(f"**ผู้แต่ง:** {book['author']}")
            st.markdown(f"**หมวดหมู่:** {book['category']}")
            st.markdown(f"**ปีที่ออก:** {book['year']}")
            st.markdown(f"**คะแนน:** ⭐ {book['rating']}/5")
            st.markdown("---")
            st.markdown(f"**เกี่ยวกับหนังสือ:**")
            st.write(book['description'])
            if book['preview']:
                st.markdown(f"[🔗 ดูตัวอย่างใน Google Books]({book['preview']})")
        
        if st.button("✖️ ปิด", key="close_detail"):
            del st.session_state.selected_book
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 หนังสือที่คล้ายกัน (Cosine Similarity)")
        
        if 'selected_idx' in st.session_state and st.session_state.selected_idx >= 0:
            similar_books = recommend_similar(
                st.session_state.selected_idx,
                st.session_state.books,
                st.session_state.doc_vectors
            )
            
            cols = st.columns(4)
            for idx, similar in enumerate(similar_books):
                with cols[idx]:
                    st.image(similar['cover'], use_column_width=True)
                    st.caption(f"**{similar['title'][:25]}**")
                    st.caption(f"{similar['author'][:20]}")
                    if st.button("ดู", key=f"sim_{similar['id']}"):
                        st.session_state.selected_book = similar
                        book_idx = next((i for i, b in enumerate(st.session_state.books) if b['id'] == similar['id']), -1)
                        st.session_state.selected_idx = book_idx
                        st.rerun()

st.markdown("---")
st.caption("🤖 AI-Enhanced IR System | TF-IDF · Vector Space Model · Cosine Similarity · Field Boosting")
