"""
Amazon Bedrock Model Invocation — Reference Implementation
-----------------------------------------------------------
Demonstrates how to invoke a foundation model in Amazon Bedrock
using the Converse API and the AWS SDK for Python (Boto3).

This artifact demonstrates:
- Amazon Bedrock Runtime
- Foundation model invocation
- Converse API
- System and user prompts
- Inference configuration
- Token usage tracking
- Basic error handling

Domain: Generative AI, Foundation Models, AWS Bedrock
"""

import boto3
import json
from botocore.exceptions import ClientError


# ------------------------------------------------------------
# AWS CONFIGURATION
# ------------------------------------------------------------

AWS_REGION = "us-east-1"

# Amazon Nova Micro is used here as an example Bedrock model.
# You can replace this with another model available in your
# AWS account and selected region.
MODEL_ID = "amazon.nova-micro-v1:0"


# bedrock-runtime is used for model inference.
# It is different from the regular "bedrock" client, which
# is used for management and configuration operations.
client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)


# ------------------------------------------------------------
# MODEL INVOCATION FUNCTION
# ------------------------------------------------------------

def invoke_model(
    user_prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3
):
    """
    Sends a prompt to an Amazon Bedrock foundation model.

    Parameters:
        user_prompt:
            The question or instruction sent by the user.

        system_prompt:
            Optional instruction that defines the model's behavior.

        max_tokens:
            Maximum number of tokens generated in the response.

        temperature:
            Controls randomness in the generated response.
            Lower values generally produce more predictable output.

    Returns:
        A tuple containing:
        - Generated model response
        - Token usage information
    """

    # Create the user message in the Converse API format.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": user_prompt
                }
            ]
        }
    ]

    # Build the request configuration.
    request = {
        "modelId": MODEL_ID,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    }

    # Add a system prompt when provided.
    if system_prompt:
        request["system"] = [
            {
                "text": system_prompt
            }
        ]

    try:
        # Send the request to Amazon Bedrock.
        response = client.converse(**request)

        # Extract the generated text.
        output_text = (
            response["output"]
            ["message"]
            ["content"][0]
            ["text"]
        )

        # Amazon Bedrock provides token usage information
        # that can be useful for monitoring and cost analysis.
        usage = response.get("usage", {})

        return output_text, usage

    except ClientError as error:
        print("Amazon Bedrock API error:")
        print(error)

        return None, None

    except Exception as error:
        print("Unexpected error:")
        print(error)

        return None, None


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    answer, token_usage = invoke_model(
        user_prompt=(
            "Explain what generative AI is and provide "
            "two common real-world applications."
        ),
        system_prompt=(
            "You are an AWS Generative AI instructor. "
            "Keep the explanation concise and beginner-friendly."
        ),
        max_tokens=300,
        temperature=0.3
    )

    if answer:

        print("\n" + "=" * 60)
        print("AMAZON BEDROCK MODEL RESPONSE")
        print("=" * 60)

        print(answer)

        print("\n" + "=" * 60)
        print("TOKEN USAGE")
        print("=" * 60)

        print(
            json.dumps(
                token_usage,
                indent=2
            )
        )
