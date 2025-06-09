CREATE TABLE IF NOT EXISTS facilities (
    id SERIAL PRIMARY KEY,
    name TEXT,
    sport TEXT,
    city TEXT,
    area TEXT,
    description TEXT
);

-- Insert sample facilities
INSERT INTO facilities (name, sport, city, area, description)
VALUES ('Cairo Football Club', 'Football', 'Cairo', 'Nasr City', 'A great place for 11-a-side football matches.');

INSERT INTO facilities (name, sport, city, area, description)
VALUES ('Zamalek Sports Center', 'Tennis', 'Cairo', 'Zamalek', 'Tennis courts available for booking');
