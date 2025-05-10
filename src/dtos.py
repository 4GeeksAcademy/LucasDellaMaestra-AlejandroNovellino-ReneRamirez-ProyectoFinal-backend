from pydantic import BaseModel


class FeaturesDto(BaseModel):
    """
    Entry features to make a prediction DTO.
    """

    # numerical features
    lead_time: float | None
    arrival_date_year: float | None
    arrival_date_week_number: float | None
    arrival_date_day_of_month: float | None
    stays_in_weekend_nights: float | None
    stays_in_week_nights: float | None
    adults: float | None
    children: float | None
    babies: float | None
    is_repeated_guest: float | None
    previous_cancellations: float | None
    previous_bookings_not_canceled: float | None
    booking_changes: float | None
    days_in_waiting_list: float | None
    adr: float | None
    required_car_parking_spaces: float | None
    total_of_special_requests: float | None

    # categorical features
    hotel: str | None
    arrival_date_month: str | None
    meal: str | None
    country: str | None
    market_segment: str | None
    distribution_channel: str | None
    is_repeated_guest: int | None
    reserved_room_type: str | None
    assigned_room_type: str | None
    deposit_type: str | None
    customer_type: str | None
