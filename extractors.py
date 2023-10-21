from typing import List, Union

from graia.ariadne import Ariadne
from graia.ariadne.event.message import GroupMessage, FriendMessage
from graia.ariadne.message.chain import MessageChain, Image


async def get_image_url(app: Ariadne, message_event: Union[GroupMessage, FriendMessage]) -> str:
    """
    Retrieves the URL of an image from the given message event.

    Args:
        app (Ariadne): The Ariadne instance.
        message_event (Union[GroupMessage, FriendMessage]): The message event.

    Returns:
        str: The URL of the image, or None if no image is found.
    """
    image_url = ""
    if Image in message_event.message_chain:
        image_url = message_event.message_chain[Image, 1][0].url
    elif message_event.quote:
        target_to_query: List[int] = []
        target_to_query.append(message_event.quote.group_id) if message_event.quote.group_id else None
        target_to_query.extend([message_event.quote.sender_id, message_event.quote.target_id])

        for target in target_to_query:
            origin_message: MessageChain = (
                await app.get_message_from_id(message_event.quote.id, target=target)
            ).message_chain
            # check if the message contains a picture
            image_url = origin_message[Image, 1][0].url if origin_message[Image, 1] else None
            break
            # FIXME cant get the quote message sent by the bot in the friend channel

    return image_url


def make_image_form_paths(img_paths: List[str]) -> List[Image]:
    return [Image(path=path) for path in img_paths]
