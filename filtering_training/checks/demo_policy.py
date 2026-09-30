from src.filtering.policy import decision_label


samples = [
    {
        "name": "서버 장애",
        "urgency": 5,
        "relevance": 5,
    },
    {
        "name": "가족 응급 상황",
        "urgency": 5,
        "relevance": 1,
    },
    {
        "name": "현재 프로젝트 PR 리뷰",
        "urgency": 3,
        "relevance": 5,
    },
    {
        "name": "README 수정",
        "urgency": 1,
        "relevance": 3,
    },
    {
        "name": "친구 잡담",
        "urgency": 1,
        "relevance": 1,
    },
]


for sample in samples:
    result = decision_label(
        sample["urgency"],
        sample["relevance"],
    )

    print(
        f'{sample["name"]}: '
        f'urgency={sample["urgency"]}, '
        f'relevance={sample["relevance"]} '
        f'-> {result}'
    )
