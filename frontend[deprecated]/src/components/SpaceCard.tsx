import React from 'react';

interface SpaceCardProps {
  title: string;
  description: string;
  icon: string;
  onClick?: () => void;
}

const SpaceCard: React.FC<SpaceCardProps> = ({ title, description, icon, onClick }) => (
  <div className="space-card" onClick={onClick}>
    <div className="text-4xl mb-4">{icon}</div>
    <h3 className="text-xl font-medium mb-2">{title}</h3>
    <p className="text-muted-foreground">{description}</p>
  </div>
);

export default SpaceCard;
