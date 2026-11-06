import { Card, Badge } from "@mantine/core";
import { ProgressBar } from "../Tremor";

interface ScoreCardProps {
  label: string;
  value: number;
  target?: number;
  max?: number;
}

export const ScoreCard = ({ label, value, target, max=1.00 }: ScoreCardProps) => {
  const passing = target === undefined ? undefined : value >= target;
  return (
    <Card withBorder className="flex-1">
      <div>
        <div className="flex justify-between text-sm mb-2">
          <p className="font-medium text-gray-900">{label}</p>
          <span className="font-medium text-gray-900 dark:text-gray-50">
            {value.toFixed(2)}
            <span className="font-normal text-gray-500">
              /{max.toFixed(2)}
            </span>
          </span>
        </div>
        <ProgressBar
          showAnimation
          className="mb-4"
          variant={passing ? "success" : "default"}
          value={value}
          max={max}
        />
        {passing !== undefined && (
          <div className="flex justify-between text-sm">
            {passing ? (
              <Badge variant="success">Passing</Badge>
            ) : (
              <Badge>Not Passing</Badge>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};
