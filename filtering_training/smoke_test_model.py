import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-1.7B"


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
        {
            "role": "user",
            "content": (
                "다음 알림의 긴급도를 1~5로 판단해줘.\n"
                "알림: 운영 서버에서 로그인 장애가 발생했습니다. "
                "즉시 확인 부탁드립니다."
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print("\nGenerating response...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\n=== Model Response ===")
    print(response)


if __name__ == "__main__":
    main()
