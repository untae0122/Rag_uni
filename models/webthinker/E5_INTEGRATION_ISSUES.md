# E5 리트리버 통합 문제점 분석 및 수정 방안

## 📋 현재 상황
`run_web_thinker_for_attack.py` 파일에서 Bing/Serper 검색 엔진을 E5 리트리버로 교체하려는 작업 중입니다.

## 🔍 발견된 문제점

### 1. **`generate_deep_web_explorer` 함수에 E5 처리 누락** ⚠️
**위치**: 346-363줄
- `generate_deep_web_explorer` 함수 내부에서 추가 검색을 수행할 때 E5 검색 엔진 처리가 없습니다
- Bing/Serper만 처리하고 있어서, Deep Web Explorer가 추가 검색을 시도하면 에러 발생

**현재 코드**:
```python
if args.search_engine == "bing":
    results = await bing_web_search_async(...)
elif args.search_engine == "serper":
    results = await google_serper_search_async(...)
else: # Should not happen
    results = {}
```

### 2. **필요한 함수들이 import되지 않음** ⚠️
**위치**: 19-28줄 (주석 처리됨)
- `fetch_page_content_async`, `extract_snippet_with_context` 함수가 주석 처리되어 있지만 코드에서 사용됨
- E5의 경우 실제로는 필요 없지만, 코드 일관성을 위해 import 필요

### 3. **`generate_deep_web_explorer`에 retriever 파라미터 없음** ⚠️
**위치**: 264-274줄
- `process_single_sequence`에서는 retriever를 받지만, `generate_deep_web_explorer`에는 전달하지 않음
- E5 검색을 수행할 수 없음

### 4. **E5 결과에 대한 URL 처리 문제** ⚠️
**위치**: 592-616줄
- E5 리트리버는 실제 URL이 아닌 `doc_id`를 반환합니다
- `fetch_page_content_async`를 호출하면 안 됩니다 (E5는 이미 전체 텍스트를 가지고 있음)
- 현재 코드는 E5 결과에 대해서도 URL fetch를 시도함

### 5. **`extract_relevant_info_e5` 함수의 키 이름 불일치** ⚠️
**위치**: 143-169줄
- E5_Retriever는 `contents` 키를 반환하지만, 코드에서는 `text`를 먼저 찾음
- `extract_relevant_info_e5`에서 `doc.get('text', doc.get('contents', ''))`로 처리하고 있어서 문제 없음
- 하지만 `extract_snippet_with_context` 호출 시 문제 발생 가능

## 🔧 수정 방안

### 수정 1: `generate_deep_web_explorer`에 retriever 파라미터 추가 및 E5 처리

```python
async def generate_deep_web_explorer(
    client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    search_query: str,
    document: str,
    search_intent: str,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    semaphore: asyncio.Semaphore,
    retriever=None,  # 추가
) -> Tuple[str, List[Dict], str]:
    # ... 기존 코드 ...
    
    # 346-363줄 수정
    if new_query in search_cache:
        results = search_cache[new_query]
    else:
        try:
            if args.search_engine == "bing":
                results = await bing_web_search_async(new_query, args.bing_subscription_key, args.bing_endpoint)
            elif args.search_engine == "serper":
                results = await google_serper_search_async(new_query, args.serper_api_key)
            elif args.search_engine == "e5":  # 추가
                if retriever:
                    results = retriever.search(new_query, k=args.top_k)
                else:
                    print("[ERROR] E5 Retriever is None!")
                    results = []
            else:
                results = {}
            search_cache[new_query] = results
        except Exception as e:
            print(f"Error during search query '{new_query}' using {args.search_engine}: {e}")
            results = {}
    
    # 358-363줄 수정
    if args.search_engine == "bing":
        relevant_info = extract_relevant_info(results)[:args.top_k]
    elif args.search_engine == "serper":
        relevant_info = extract_relevant_info_serper(results)[:args.top_k]
    elif args.search_engine == "e5":  # 추가
        relevant_info = extract_relevant_info_e5(results, top_k=args.top_k)
    else:
        relevant_info = []
```

### 수정 2: `process_single_sequence`에서 E5일 때 URL fetch 스킵

```python
# 592-616줄 수정
# Process documents
urls_to_fetch = []
if args.search_engine != "e5":  # E5는 URL fetch 불필요
    for doc_info in relevant_info:
        url = doc_info['url']
        if url not in url_cache:
            urls_to_fetch.append(url)

if urls_to_fetch:
    # ... 기존 fetch 로직 ...
```

### 수정 3: E5 결과에 대한 `extract_snippet_with_context` 호출 수정

```python
# 618-635줄 수정
for doc_info in relevant_info:
    url = doc_info['url']
    if args.search_engine == "e5":
        # E5는 이미 전체 텍스트를 가지고 있음
        doc_info['page_info'] = doc_info.get('page_info', doc_info.get('contents', ''))
    else:
        if url not in url_cache:
            raw_content = ""
        else:
            raw_content = url_cache[url]
            is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=2000)
        
        has_error = any(indicator.lower() in raw_content.lower() for indicator in error_indicators) or raw_content == ""
        if has_error:
            doc_info['page_info'] = "Can not fetch the page content."
        else:
            doc_info['page_info'] = raw_content
```

### 수정 4: `generate_deep_web_explorer` 호출 시 retriever 전달

```python
# 651줄 수정
analysis, explorer_prompt = await generate_deep_web_explorer(
    client=client,
    aux_client=aux_client,
    search_query=search_query,
    search_intent=search_intent,
    document=formatted_documents,
    args=args,
    search_cache=search_cache,
    url_cache=url_cache,
    semaphore=semaphore,
    retriever=retriever,  # 추가
)
```

### 수정 5: `extract_relevant_info_e5` 함수 개선

```python
def extract_relevant_info_e5(results, top_k=5):
    """
    E5 검색 결과를 WebThinker 포맷으로 변환
    results: E5_Retriever.search()가 반환한 리스트
    """
    relevant_info = []
    for i, doc in enumerate(results[:top_k]):
        # E5_Retriever는 {'id', 'title', 'contents', 'score', 'is_poisoned'} 반환
        title = doc.get('title', f"Document {i+1}")
        text = doc.get('contents', '')  # 'contents'가 메인 키
        doc_id = doc.get('id', f"doc_id_{i}")
        
        relevant_info.append({
            'title': title,
            'url': doc_id,  # 실제 URL이 아니라 doc_id
            'snippet': text[:500] + "..." if len(text) > 500 else text,
            'page_info': text,  # 전체 텍스트
            'is_poisoned': doc.get('is_poisoned', False)  # 추가 정보
        })
    return relevant_info
```

## 📝 추가 고려사항

1. **E5는 링크 클릭 기능 불필요**: E5는 이미 전체 문서를 가지고 있으므로 `<|begin_click_link|>` 기능은 의미가 없습니다. 하지만 코드 호환성을 위해 남겨둘 수 있습니다.

2. **캐싱**: E5 검색 결과도 캐싱되므로 문제없습니다.

3. **에러 처리**: E5 검색 실패 시 빈 리스트 반환하도록 처리되어 있습니다.

## ✅ 수정 우선순위

1. **높음**: `generate_deep_web_explorer`에 retriever 파라미터 추가 및 E5 처리
2. **높음**: `process_single_sequence`에서 E5일 때 URL fetch 스킵
3. **중간**: E5 결과 처리 로직 개선
4. **낮음**: import 정리 (E5 사용 시 불필요하지만 일관성 위해)
