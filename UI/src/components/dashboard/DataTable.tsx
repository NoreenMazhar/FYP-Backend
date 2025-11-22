import { Card } from "../ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import { Badge } from "../ui/badge";
import React from "react";
const recentOrders = [
  {
    id: "ORD-001",
    customer: "John Smith",
    product: "Wireless Headphones",
    amount: "$199.99",
    status: "Completed",
  },
  {
    id: "ORD-002",
    customer: "Emma Wilson",
    product: "Smart Watch",
    amount: "$349.99",
    status: "Processing",
  },
  {
    id: "ORD-003",
    customer: "Michael Brown",
    product: "Laptop Stand",
    amount: "$79.99",
    status: "Completed",
  },
  {
    id: "ORD-004",
    customer: "Sarah Davis",
    product: "USB-C Cable",
    amount: "$24.99",
    status: "Shipped",
  },
  {
    id: "ORD-005",
    customer: "James Johnson",
    product: "Keyboard",
    amount: "$129.99",
    status: "Processing",
  },
  {
    id: "ORD-006",
    customer: "Lisa Anderson",
    product: "Mouse Pad",
    amount: "$19.99",
    status: "Completed",
  },
  {
    id: "ORD-007",
    customer: "David Martinez",
    product: "Monitor",
    amount: "$449.99",
    status: "Pending",
  },
];

export function DataTable() {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "Completed":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case "Processing":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case "Shipped":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200";
      case "Pending":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
    }
  };

  return (
    <Card className="p-6">
      <div className="mb-4">
        <h3>Recent Orders</h3>
        <p className="text-muted-foreground">
          Latest transactions from your store
        </p>
      </div>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Order ID</TableHead>
              <TableHead>Customer</TableHead>
              <TableHead>Product</TableHead>
              <TableHead>Amount</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {recentOrders.map((order) => (
              <TableRow
                key={order.id}
                className="hover:bg-muted/50 transition-colors"
              >
                <TableCell>{order.id}</TableCell>
                <TableCell>{order.customer}</TableCell>
                <TableCell>{order.product}</TableCell>
                <TableCell>{order.amount}</TableCell>
                <TableCell>
                  <Badge
                    variant="outline"
                    className={getStatusColor(order.status)}
                  >
                    {order.status}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}
