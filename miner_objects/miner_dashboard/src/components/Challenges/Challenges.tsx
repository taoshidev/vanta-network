import { Title } from "@mantine/core";
import { isNil } from "lodash";
import { Fragment } from "react";
import { Statistics as StatisticsType } from "../../types";
import { ScoreCard } from "../ScoreCard";

interface ChallengesProps {
  statistics: StatisticsType;
}

export const Challenges = ({ statistics }: ChallengesProps) => {
  const { challengeperiod, scores } = statistics.data[0];
  const { status } = challengeperiod;

  // if anything is in challenge period show element
  const isInChallenge = status === "testing";
  if (!isInChallenge) return null;

  const { omega, calmar, return:returnScore, sharpe, sortino, statistical_confidence } = scores

  const scoreData = [
    { label: "Omega", score: omega },
    { label: "Sharpe Ratio", score: sharpe },
    { label: "Sortino", score: sortino },
    { label: "Statistical Confidence", score: statistical_confidence },
    { label: "Calmar", score: calmar },
    { label: "Return", score: returnScore },
  ];

  return (
    <Fragment>
      {isInChallenge && (
        <div className="mb-8">
          <Title order={3} mb="sm">
            Challenge Period
          </Title>
          <div className="flex gap-4">
            {scoreData.map(({ label, score }) => {
              if (!isNil(score)) {
                return (
                  <ScoreCard
                    key={label}
                    label={label}
                    value={score.percentile}
                  />
                );
              }
              return null;
            })}
          </div>
        </div>
      )}
    </Fragment>
  );
};
