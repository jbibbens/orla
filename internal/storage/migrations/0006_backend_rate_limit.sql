-- +goose Up
ALTER TABLE backends
    ADD COLUMN rate_per_second DOUBLE PRECISION;

-- +goose Down
ALTER TABLE backends DROP COLUMN rate_per_second;
